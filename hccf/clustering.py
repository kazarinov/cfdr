# -*- coding: utf-8 -*-
import os
import itertools
import logging
import pickle
import copy
import multiprocessing

from hccf.utils.tools import parse_vw_line, compose_vw_line
from hccf.utils.mathematics import loglikelihood


log = logging.getLogger(__name__)


class Node(object):
    """
    Класс представляющий узел бинарного дерева иерархии.
    """
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def show(self, depth=0):
        """
        Метод возвращяющий строку для печати дерева на экране

        :param depth: отсуп узла при отображении в консоли
        :return: строку для печати в консоли
        """
        ret = ""

        # Print right branch
        if self.right is not None:
            ret += self.right.show(depth + 1)

        # Print own value
        ret += "\n" + ("    "*depth) + str(self.value)

        # Print left branch
        if self.left is not None:
            ret += self.left.show(depth + 1)

        return ret

    def get_leaves_parents(self, parents=None):
        """
        Метод возвращяющий список родителей данного узла

        :param parents: list родители узла предыдущего уровня
        :return: list список родителей данного узла
        """
        result = {}

        if parents is None:
            parents = []

        parents_copy = parents[:]
        if self.left is None and self.right is None:
            result[self.value] = parents_copy
        else:
            parents_copy.append(self.value)
            if self.left is not None:
                result.update(self.left.get_leaves_parents(parents_copy))
            if self.right is not None:
                result.update(self.right.get_leaves_parents(parents_copy))

        return result

    def to_gml(self):
        """
        Метод возвращает GML отображение графа
        """
        nodes, edges = self._gml()
        nodes[self.value] = self.value

        res = "graph\n[\n"
        for node_index, node_label in enumerate(nodes.keys()):
            nodes[node_label] = node_index
            res += ('  node\n  [\n    id %i\n    label "%s"\n  ]\n' %
                    (node_index, node_label))
        for edge in edges:
            res += ('  edge\n  [\n    source %i\n    target %i\n  ]\n' %
                    (nodes[edge["source"]], nodes[edge["target"]]))
        res += "]\n"
        return res

    def _gml(self):
        nodes = {}
        edges = []

        if self.left is not None:
            nodes[self.left.value] = self.left.value
            edges.append({
                'source': self.value,
                'target': self.left.value,
            })
            left_nodes, left_edges = self.left._gml()
            nodes.update(left_nodes)
            edges += left_edges

        if self.right is not None:
            nodes[self.right.value] = self.right.value
            edges.append({
                'source': self.value,
                'target': self.right.value,
            })
            right_nodes, right_edges = self.right._gml()
            nodes.update(right_nodes)
            edges += right_edges

        return nodes, edges


def calculate_pair_ll(feature_value_pair):
    """
    Функция, вычисляющая likelihood пары значений категориального фактора

    :param feature_value_pair: пара значений категориального фактора
    :return: float значение likelihood
    """
    ll_pair = 0
    tmp_feature_value_data = {}
    for feature_value_pair_index in feature_value_pair:
        for feature_values, (shows, clicks) in FeatureClustering.feature_data[feature_value_pair_index].iteritems():
            ll_pair -= loglikelihood(shows, clicks)

            tmp_feature_values = list(feature_values)
            tmp_feature_values[FeatureClustering.feature_index] = None
            tmp_feature_values = tuple(tmp_feature_values)

            if tmp_feature_values not in tmp_feature_value_data.keys():
                tmp_feature_value_data[tmp_feature_values] = (shows, clicks)
            else:
                shows_tmp, clicks_tmp = tmp_feature_value_data[tmp_feature_values]
                tmp_feature_value_data[tmp_feature_values] = (shows + shows_tmp, clicks + clicks_tmp)

    for feature_values, (shows, clicks) in tmp_feature_value_data.iteritems():
        ll_pair += loglikelihood(shows, clicks)

    return ll_pair


class FeatureClustering(object):
    """
    Класс для построения иерархической кластеризации значений категориальных факторов
    """
    feature_data = None
    feature_index = None

    def __init__(self, processes=1):
        """
        :param processes: количество процессов для параллельного режима, если =1, то однопоточный режим
        """
        if not isinstance(processes, int) or processes < 1:
            raise ValueError('‘processes’ param has to be positive integer')

        self.processes = processes

        self.features_mapping = None
        self.features = None

        self.data = {}
        self.shows_count = 0
        self.clicks_count = 0
        self.trees = None

    def _preprocess_log(self, logfile, slice_features=None, parser=None):
        """
        Метод обрабатывающий файл и инициализующий
         - словарь self.data, ключами которого являются пары значений в событиях,
            а значением количество кликов и показов
         - словарь self.features, ключи которого порядковые индексы факторов в файле,
            а значения – возможные значения данного фактора

        :param logfile: string файл для обработки
        :param slice_features: названия факторов, по которым происходит сред
        :param parser: название формата для обработка файла (vw | libffm)
        :return:
        """
        if parser is None:
            parser = parse_vw_line

        for line in open(logfile):
            example_params = parser(line)
            if self.features is None:
                if slice_features is not None:
                    self.features = [set() for i in xrange(len(slice_features))]
                    self.features_mapping = dict([(feature_name, feature_index)
                                                  for feature_index, feature_name in
                                                  enumerate(slice_features)])
                else:
                    self.features = [set() for i in xrange(len(example_params['features']))]
                    self.features_mapping = dict([(feature_name, feature_index)
                                                  for feature_index, (feature_name, feature_value) in
                                                  enumerate(example_params['features'])])

            for feature_name, feature_value in example_params['features']:
                feature_index = self.features_mapping.get(feature_name)
                if feature_index is not None:
                    self.features[feature_index].add(feature_value)

            if slice_features is not None:
                feature_example_dict = dict(example_params['features'])
                example_values = tuple((feature_example_dict[feature_name] for feature_name in slice_features))
            else:
                example_values = tuple((feature_value for feature_name, feature_value in example_params['features']))

            if example_values not in self.data:
                self.data[example_values] = (0, 0)

            shows, clicks = self.data[example_values]
            shows += 1
            self.shows_count += 1
            if example_params['label'] > 0:
                clicks += 1
                self.clicks_count += 1
            self.data[example_values] = (shows, clicks)

    def gml_graph(self, feature):
        """
        Метод возвращает отображение иерархии значений категориального фактора в GML формате
        :param feature: название фактора
        :return: string
        """
        if feature not in self.trees.keys():
            raise ValueError('no hierarchy for feature `%s`' % feature)
        return self.trees[feature].to_gml()

    def convert_log(self, input_logfile, output_logfile, output_params, parser=None, composer=None):
        """
        Метод, преобразующий исходных набов данных в набор данных с дополнительными факторами после кластеризации

        :param input_logfile: string входной файл
        :param output_logfile: string выходной файл
        :param output_params: dict параметры дополнительных факторов
             output_params: {
                'mode': <mode>,
                'levels': 5
             }

            <mode>:
             * full_tree - полное дерево до корня с листьями
             * tree_without_leaves - дерево без листьев
             * part_tree

        :param parser: func функция-парсер исходного набора данных
        :param composer: func функция-композитор выходного набора
        """
        if parser is None:
            parser = parse_vw_line

        if composer is None:
            composer = compose_vw_line

        output_file = open(output_logfile, 'w+')
        output_mode = output_params.get('mode', 'full_tree')
        feature_mapping = {}

        for feature_name, feature_tree in self.trees.iteritems():
            feature_mapping[feature_name] = feature_tree.get_leaves_parents()

        for line in open(input_logfile):
            example_params = parser(line)
            features = []
            for feature_name, feature_value in example_params['features']:
                if feature_name in self.trees.keys():
                    feature_value_tree = feature_mapping[feature_name].get(feature_value, [])
                    if output_mode == 'full_tree':
                        features.append((feature_name, feature_value))
                    elif output_mode == 'tree_without_leaves':
                        pass
                    elif output_mode == 'part_tree':
                        features.append((feature_name, feature_value))
                        feature_value_tree = feature_value_tree[len(feature_value_tree) - output_params['levels']: len(feature_value_tree)]
                    features.append((feature_name + '_tree', feature_value_tree))
                else:
                    features.append((feature_name, feature_value))

            output_file.write(composer(example_params['label'], features) + '\n')

    def cluster(self, input_logfile, tree_features, slice_features=None, parser=None):
        """
        Метод для построения иерархии значений категориальных факторов

        :param input_logfile: string входной файл
        :param tree_features: list список названий категориальных факторов для кластеризации
        :param slice_features: list список "разрезов" для построения иерархии
        :param parser: func функция-парсер исходного набора данных
        :return: dict деревья (иерархии кластеров) для каждого категориального фактора, переданного в tree_features
        """
        if not os.path.isfile(input_logfile):
            raise ValueError('no input file `%s` found' % input_logfile)
        elif len(tree_features) <= 0:
            raise ValueError('`tree_features` param was not passed')

        features = copy.copy(tree_features)
        if slice_features is not None:
            features += slice_features
        features = list(set(features))
        self._preprocess_log(input_logfile, slice_features=features, parser=parser)

        feature_names = self.features_mapping.keys()
        for feature_name in tree_features:
            if feature_name not in feature_names:
                raise ValueError('feature `%s` was not found in input log' % feature_name)

        self.trees = {}
        for feature_name in tree_features:
            if self.processes > 1:
                self.trees[feature_name] = self._cluster_feature_parallel(feature_name)
            else:
                self.trees[feature_name] = self._cluster_feature(feature_name)
        return self.trees

    def _join_feature_value_pair(self, feature_index, feature_value_pair, feature_data):
        """
        Метод для объединения значений в кластер

        :param feature_index: int
        :param feature_value_pair: tuple (feature_value1, feature_value2)
        :param feature_data: dict
        :return: dict
        """
        tmp_feature_value_data = {}
        for feature_value_pair_index in feature_value_pair:
            for feature_values, (shows, clicks) in feature_data[feature_value_pair_index].iteritems():
                tmp_feature_values = list(feature_values)
                tmp_feature_values[feature_index] = None
                tmp_feature_values = tuple(tmp_feature_values)

                if tmp_feature_values not in tmp_feature_value_data.keys():
                    tmp_feature_value_data[tmp_feature_values] = (shows, clicks)
                else:
                    shows_tmp, clicks_tmp = tmp_feature_value_data[tmp_feature_values]
                    tmp_feature_value_data[tmp_feature_values] = (shows + shows_tmp, clicks + clicks_tmp)
        return tmp_feature_value_data

    def _cluster_feature_parallel(self, feature_name):
        """
        Метод для построения иерархии значений категориального фактора <feature_name> в многопоточном режиме
        Количество процессов задается в конструкторе (параметр processes)

        :param feature_name: string название категориального фактора
        :return: Node дерево (иерархии кластеров) для категориального фактора
        """
        feature_index = FeatureClustering.feature_index = self.features_mapping[feature_name]
        FeatureClustering.feature_data = {}
        for feature_value, data in self.data.iteritems():
            feature_value_data = FeatureClustering.feature_data.setdefault(feature_value[feature_index], {})
            feature_value_data[feature_value] = data

        current_extra_node_index = 1
        extra_nodes = {}

        while len(FeatureClustering.feature_data.keys()) > 1:
            pool = multiprocessing.Pool(self.processes)
            min_ll_pair_delta = float("inf")
            min_pair = None

            feature_value_pairs = list(itertools.combinations(FeatureClustering.feature_data.keys(), r=2))
            feature_value_ll = pool.map(calculate_pair_ll, feature_value_pairs)

            for i in xrange(len(feature_value_ll)):
                if feature_value_ll[i] <= min_ll_pair_delta:
                    min_pair = feature_value_pairs[i]
                    min_ll_pair_delta = feature_value_ll[i]

            tmp_feature_value_data = self._join_feature_value_pair(
                feature_index,
                min_pair,
                FeatureClustering.feature_data,
            )

            for feature_value_pair_index in min_pair:
                del FeatureClustering.feature_data[feature_value_pair_index]

            current_extra_node_label = str(current_extra_node_index) + '*'
            FeatureClustering.feature_data[current_extra_node_label] = tmp_feature_value_data

            extra_nodes[current_extra_node_label] = Node(
                current_extra_node_label,
                left=extra_nodes.get(min_pair[0], Node(min_pair[0])),
                right=extra_nodes.get(min_pair[1], Node(min_pair[1])),
            )
            current_extra_node_index += 1
            log.debug('min_pair %s, extra_node_index=%s, min_ll_pair=%s', min_pair, current_extra_node_index, min_ll_pair_delta)
            pool.close()
            pool.join()

        tree = extra_nodes[FeatureClustering.feature_data.keys()[0]]
        return tree

    def _cluster_feature(self, feature_name):
        """
        Метод для построения иерархии значений категориального фактора <feature_name> в однопоточном режиме

        :param feature_name: string название категориального фактора
        :return: Node дерево (иерархии кластеров) для категориального фактора
        """
        feature_index = self.features_mapping[feature_name]
        feature_data = {}
        # Для оптимизации перекладываем в словарь,
        # ключами которого являются значения кластеризуемого фактора
        for feature_value, data in self.data.iteritems():
            feature_value_data = feature_data.setdefault(feature_value[feature_index], {})
            feature_value_data[feature_value] = data

        current_extra_node_index = 1
        extra_nodes = {}

        while len(feature_data.keys()) > 1:
            # Инициализируем параметры для выбора пары значений для объединения
            min_ll_pair = float("inf")
            min_pair = None
            min_tmp_feature_value_data = None

            # Строим все пары значений кластеризуемого фактора
            for feature_value_pair in itertools.combinations(feature_data.keys(), r=2):
                ll_pair = 0
                tmp_feature_value_data = {}
                for feature_value_pair_index in feature_value_pair:
                    for feature_values, (shows, clicks) in feature_data[feature_value_pair_index].iteritems():
                        ll_pair -= loglikelihood(shows, clicks)

                        tmp_feature_values = list(feature_values)
                        tmp_feature_values[feature_index] = None
                        tmp_feature_values = tuple(tmp_feature_values)

                        if tmp_feature_values not in tmp_feature_value_data.keys():
                            tmp_feature_value_data[tmp_feature_values] = (shows, clicks)
                        else:
                            shows_tmp, clicks_tmp = tmp_feature_value_data[tmp_feature_values]
                            tmp_feature_value_data[tmp_feature_values] = (shows + shows_tmp, clicks + clicks_tmp)

                for feature_values, (shows, clicks) in tmp_feature_value_data.iteritems():
                    ll_pair += loglikelihood(shows, clicks)

                if ll_pair <= min_ll_pair:
                    min_ll_pair = ll_pair
                    min_pair = feature_value_pair
                    min_tmp_feature_value_data = tmp_feature_value_data.copy()

            for feature_value_pair_index in min_pair:
                del feature_data[feature_value_pair_index]

            current_extra_node_label = str(current_extra_node_index) + '*'
            feature_data[current_extra_node_label] = min_tmp_feature_value_data

            # Добавляем новый узел девера (иерархия кластеров)
            extra_nodes[current_extra_node_label] = Node(
                current_extra_node_label,
                left=extra_nodes.get(min_pair[0], Node(min_pair[0])),
                right=extra_nodes.get(min_pair[1], Node(min_pair[1])),
            )
            current_extra_node_index += 1
            log.debug('min_pair %s, extra_node_index=%s, min_ll_pair=%s', min_pair, current_extra_node_index, min_ll_pair)

        tree = extra_nodes[feature_data.keys()[0]]
        return tree

    @staticmethod
    def load(filename):
        """
        Метод загружает модель иерархической кластеризации из файла
        :return: FeatureClustering
        """
        f = file(filename)
        model = pickle.loads(f.read())
        f.close()
        return model

    def save(self, filename):
        """
        Метод сохраняет модель иерархической кластеризации в файл
        """
        f = file(filename, 'w+')
        f.write(pickle.dumps(self))
        f.close()
