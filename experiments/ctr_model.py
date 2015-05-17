# -*- coding: utf-8 -*-
import itertools
import pickle

import numpy as np
from fabric.api import local

from hccf.utils.helpers import silentremove
from hccf.utils.mathematics import (
    sigmoid,
    loglikelihood,
)
from hccf.utils.tools import (
    compose_vw_line,
    compose_libffm_line,
)


class CTRModel(object):
    def __init__(self, features_config, free_coef=1, lam=20):
        """
        Генерирует искусственный лог кликов
        :param features_config: dict описание категориальных фичей и их параметры распределения
          {
            'cat1': {
                'count': 1000,
                'loc': 0.0,
                'scale': 0.5,
                'type': 'tree',
            },
            'cat2': {
                'count': 1000,
                'loc': 0.0,
                'scale': 0.5,
                'type': 'tree',
            },
            'cat1xcat2': {
                'loc': 0.0,
                'scale': 0.1,
                'parts': ['cat1', 'cat2'],
            }
          }
          loc - мат. ожидание
          scale - стандартное отклонение

        :param free_coef: float свободный коэффициент в логистической модели
        :param lmbd: float параметр "лямбда" в распределении Пуассона для генерации числа показов
        :return:
        """
        self.features_config = features_config
        self.free_coef = free_coef
        self.lam = lam

        self.shows = None
        self.clicks = None
        self.coefficients = {}

    def init(self):
        self._generate_coefficients()
        self._generate_shows_clicks()

    def _generate_tree_coeffiecients(self, loc, scale, count):
        coefficients = np.zeros(count)

        levels_number = int(round(np.log2(count))) + 1
        loc = float(loc) / levels_number
        scale = np.sqrt((float(scale)**2) / levels_number)
        for level in xrange(levels_number):
            step = 2 ** level
            add_coef = 0
            for i in xrange(count):
                if i % step == 0:
                    add_coef = np.random.normal(loc, scale)
                coefficients[i] += add_coef
        return coefficients

    def _generate_coefficients(self):
        for feature_name, feature_params in self.features_config.items():
            if feature_params.get('type') == 'tree':
                if feature_params.get('parts'):
                    tree_coefficients = []
                    sizes = []
                    part_scale = np.sqrt((feature_params['scale']**2) / len(feature_params['parts']))
                    for feature_part_name in feature_params['parts']:
                        if self.features_config[feature_part_name].get('type') == 'tree':
                            tree_coefficients.append(self._generate_tree_coeffiecients(
                                loc=feature_params['loc'],
                                scale=part_scale,
                                count=self.features_config[feature_part_name]['count'],
                            ))
                        else:
                            tree_coefficients.append(np.random.normal(
                                self.features_config[feature_part_name]['loc'],
                                self.features_config[feature_part_name]['scale'],
                                self.features_config[feature_part_name]['count'],
                            ))
                        sizes.append(self.features_config[feature_part_name]['count'])

                    self.coefficients[feature_name] = np.zeros(sizes)
                    for feature_part_values in itertools.product(*[xrange(count) for count in sizes]):
                        for feature_part_index, feature_part_value in enumerate(feature_part_values):
                            self.coefficients[feature_name][feature_part_values] += tree_coefficients[feature_part_index][feature_part_value]
                else:
                    self.coefficients[feature_name] = self._generate_tree_coeffiecients(
                        loc=feature_params['loc'],
                        scale=feature_params['scale'],
                        count=feature_params['count'],
                    )
            else:
                if feature_params.get('parts'):
                    count = [self.features_config[feature_part_name]['count']
                             for feature_part_name in feature_params['parts']]
                else:
                    count = feature_params['count']

                self.coefficients[feature_name] = np.random.normal(
                    feature_params['loc'],
                    feature_params['scale'],
                    count,
                )

    def _generate_shows_clicks(self):
        features = self._get_single_features()
        sizes = [self.features_config[feature]['count'] for feature in features]
        self.shows = np.random.poisson(self.lam, size=sizes)
        self.clicks = np.zeros(sizes)

        for feature_values in itertools.product(*[xrange(count) for count in sizes]):
            shows_for_features = self.shows[feature_values]
            feature_values_ctr = self.ctr(**dict(itertools.izip(features, feature_values)))
            self.clicks[feature_values] = shows_for_features * feature_values_ctr

    def _get_single_features(self):
        return [feature_name
                for feature_name, feature_params in self.features_config.items()
                if not feature_params.get('parts')]

    def _get_features_sizes(self):
        features = self._get_single_features()
        return [self.features_config[feature]['count'] for feature in features]

    def _get_coefficient(self, feature_name, feature_value=None, **values):
        if feature_value is not None:
            return self.coefficients[feature_name][feature_value]
        else:
            if not self.features_config[feature_name].get('parts'):
                raise ValueError('Not partial feature %s' % feature_name)

            if sorted(values.keys()) != sorted(self.features_config[feature_name]['parts']):
                raise ValueError('Not all feature values are given %s' % values.keys())

            feature_coefficient = self.coefficients[feature_name]
            for feature_part_name in self.features_config[feature_name]['parts']:
                feature_coefficient = feature_coefficient[values[feature_part_name]]
            return feature_coefficient

    def ctr(self, **features):
        if sorted(features.keys()) != sorted(self._get_single_features()):
            raise ValueError('Not all features are given. %s' % features.keys())

        linear_model = self.free_coef
        for feature_name, feature_params in self.features_config.items():
            if feature_params.get('parts'):
                feature_values = {}
                for feature_part_name in feature_params['parts']:
                    feature_values[feature_part_name] = features[feature_part_name]
                linear_model += self._get_coefficient(feature_name, **feature_values)
            else:
                linear_model += self._get_coefficient(feature_name, features[feature_name])

        return sigmoid(linear_model)

    def generate_log(self, filename, format='csv', train_percentage=1.0, train_length=None, test_length=None):
        tmp_filename = filename + '.' + format + '.tmp'
        train_filename = filename + '.train.' + format
        test_filename = filename + '.test.' + format
        shuffle_filename = filename + '.shuffle.' + format

        features = self._get_single_features()
        sizes = [self.features_config[feature]['count'] for feature in features]

        tmplogfile = open(tmp_filename, 'w+')
        # generate clicklog by ctr model
        for feature_values in itertools.product(*[xrange(count) for count in sizes]):
            shows_for_features = self.shows
            for feature_value in feature_values:
                shows_for_features = shows_for_features[feature_value]

            feature_values_ctr = self.ctr(**dict(itertools.izip(features, feature_values)))
            clicks_for_features = int(round(shows_for_features * feature_values_ctr))

            for i in xrange(shows_for_features):
                is_click = i < clicks_for_features
                if format == 'csv':
                    tmplogfile.write(('%s,' % int(is_click)) + ','.join(map(str, feature_values)) + '\n')
                elif format == 'vw':
                    tmplogfile.write(compose_vw_line(int(is_click), zip(features, map(str, feature_values))) + '\n')
                elif format == 'libffm':
                    tmplogfile.write(compose_libffm_line(is_click, zip(features, map(str, feature_values))) + '\n')

        tmplogfile.close()

        # shuffle dataset
        local('shuf %s > %s' % (tmp_filename, shuffle_filename))

        lines_number = np.sum(self.shows)
        if train_length:
            train_lines_number = train_length
        elif train_percentage:
            train_lines_number = lines_number * train_percentage

        # split dataset int training and testing
        train_file = open(train_filename, 'w+')
        test_file = open(test_filename, 'w+')

        if format == 'csv':
            headline = 'click,' + ','.join(features) + '\n'
            train_file.write(headline)
            test_file.write(headline)

        line_index = 0
        for line in open(shuffle_filename):
            if line_index < train_lines_number:
                train_file.write(line)
            else:
                if test_length and (line_index - train_lines_number) >= test_length:
                    break
                test_file.write(line)
            line_index += 1

        train_file.close()
        test_file.close()

        # cleanup
        silentremove(tmp_filename)
        silentremove(shuffle_filename)


    @staticmethod
    def load(filename):
        f = file(filename)
        ctr_model = pickle.loads(f.read())
        f.close()
        return ctr_model

    def save(self, filename):
        f = file(filename, 'w+')
        f.write(pickle.dumps(self))
        f.close()

    def loglikelihood(self):
        ll = 0

        sizes = self._get_features_sizes()

        for feature_values in itertools.product(*[xrange(count) for count in sizes]):
            s = self.shows[feature_values]
            c = self.clicks[feature_values]
            ll += loglikelihood(s, c)

        return ll / np.sum(self.shows)

    def loglikelihood0(self):
        shows_n = np.sum(self.shows)
        clicks_n = np.sum(self.clicks)
        return loglikelihood(shows_n, clicks_n) / shows_n

    def likelihood_ratio(self):
        return self.loglikelihood0() - self.loglikelihood()
