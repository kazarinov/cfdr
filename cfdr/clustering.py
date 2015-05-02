# -*- coding: utf-8 -*-
import math
import itertools
import logging
import multiprocessing

from cfdr.utils.vw import parse_vw_line, compose_vw_line
from cfdr.utils.mathematics import loglikelihood


log = logging.getLogger(__name__)


class Node(object):
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def show(self, depth=0):
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


class FeatureClustering(object):
    def __init__(self):
        self.features_mapping = None
        self.features = None

        self.data = {}
        self.shows_count = 0
        self.clicks_count = 0
        self.trees = None

    def preprocess_log(self, logfile, slice_features=None):
        for line in open(logfile):
            example_params = parse_vw_line(line)
            if self.features is None:
                self.features = [set() for i in xrange(len(example_params['features']))]
                self.features_mapping = dict([(feature_name, feature_index)
                                              for feature_index, (feature_name, feature_value) in
                                              enumerate(example_params['features'])])

            for feature_index, feature in enumerate(example_params['features']):
                feature_name, feature_value = feature
                self.features[feature_index].add(feature_value)

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

    def convert_log(self, input_logfile, output_logfile, output_params):
        output_file = open(output_logfile, 'w+')
        output_mode = output_params.get('mode', 'full_tree')
        feature_mapping = {}

        for feature_name, feature_tree in self.trees.iteritems():
            feature_mapping[feature_name] = feature_tree.get_leaves_parents()

        for line in open(input_logfile):
            example_params = parse_vw_line(line)
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

            output_file.write(compose_vw_line(example_params['label'], features) + '\n')

    def loglikelihood0(self):
        return loglikelihood(self.shows_count, self.clicks_count)

    def loglikelihood(self, feature_data):
        ll = 0
        for feature_value, feature_value_data in feature_data.iteritems():
            for _, (shows, clicks) in feature_value_data.iteritems():
                ll += loglikelihood(shows, clicks)
        return ll

    def cluster(self, input_logfile, tree_features, slice_features=None):
        """
        output_params: {
            'mode': <mode>,
            'levels': 5
        }

        <mode>:
         * full_tree - полное дерево до корня с листьями
         * tree_without_leaves - дерево без листьев
         * part_tree
        """
        self.preprocess_log(input_logfile, slice_features=slice_features)

        self.trees = {}
        for feature_nane in tree_features:
            self.trees[feature_nane] = self.cluster_feature(feature_nane)
        return self.trees

    def cluster_feature(self, feature_name):
        feature_index = self.features_mapping[feature_name]
        feature_data = {}
        for feature_value, data in self.data.iteritems():
            feature_value_data = feature_data.setdefault(feature_value[feature_index], {})
            feature_value_data[feature_value] = data

        ll = self.loglikelihood(feature_data)
        current_extra_node_index = 1
        extra_nodes = {}

        while len(feature_data.keys()) > 1:
            min_ll_pair = float("inf")
            min_pair = None
            min_tmp_feature_value_data = None
            for feature_value_pair in itertools.combinations(feature_data.keys(), r=2):
                ll_pair = ll
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

            feature_data[current_extra_node_index] = min_tmp_feature_value_data
            extra_nodes[current_extra_node_index] = Node(
                current_extra_node_index,
                left=extra_nodes.get(min_pair[0], Node(min_pair[0])),
                right=extra_nodes.get(min_pair[1], Node(min_pair[1])),
            )
            current_extra_node_index += 1
            ll = min_ll_pair
            log.debug('min_pair %s, extra_node_index=%s, min_ll_pair=%s', min_pair, current_extra_node_index, min_ll_pair)

        tree = extra_nodes[feature_data.keys()[0]]
        return tree


"""
Алгоритм кластеризации

Строим таблицу по каждому значению категориального фактора

Если некоторые факторы real-value, то нужно их кластеризовать.
Или просто разделить область значений данной фичи на N равных участков

Профиль CTR'ов по одному фактору отличается от другого. Может быть нужно усреднять?
Или же нужно брать во всех разрезах


1. Определяем LL0 тренировочного датасета (средний CTR)
2. Можем посчитать LL тренировочного датасета
3. Формируем

"""


def main():
    fc = FeatureClustering()
    fc.cluster('clicklog2.train.vw', ['cat1'])
    fc.convert_log('clicklog2.train.vw', 'clicklog2-output.train.vw',  {'mode': 'full_tree'})
    fc.convert_log('clicklog2.test.vw', 'clicklog2-output.test.vw',  {'mode': 'full_tree'})
    # fc.preprocess_log('clicklog2.train.vw')
    # fc.cluster_feature('cat1')
    # print 'cat1', len(fc.features[fc.features_mapping['cat1']])
    # print 'cat2', len(fc.features[fc.features_mapping['cat1']])
    # print 'data', len(fc.data.keys())
    # print 'data', fc.data


if __name__ == '__main__':
    main()
    # tree = Node(1, Node(2, Node(3), Node(4)), Node(5, Node(6), Node(7)))
    # print tree.show()
    #
    # for fv, flist in tree.get_leaves_parents().items():
    #     print fv, flist
