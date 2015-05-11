# -*- coding: utf-8 -*-
import logging

from hccf.utils.tools import parse_vw_line
from hccf.experiments.models.base import BaseClassifier


log = logging.getLogger(__name__)


class HistoricClassifier(BaseClassifier):
    def __init__(self):
        self.features_mapping = None
        self.features = None

        self.data = {}
        self.shows_count = 0
        self.clicks_count = 0

    def train(self, train_filename):
        for line in open(train_filename):
            example_params = parse_vw_line(line)
            if self.features is None:
                self.features = [dict() for i in xrange(len(example_params['features']))]
                self.features_mapping = dict([(feature_name, feature_index)
                                              for feature_index, (feature_name, feature_value) in
                                              enumerate(example_params['features'])])

            for feature_index, feature in enumerate(example_params['features']):
                feature_name, feature_value = feature
                feature_value_shows, feature_value_clicks = self.features[feature_index].setdefault(feature_value, (0, 0))

                feature_value_shows += 1
                if example_params['label'] > 0:
                    feature_value_clicks += 1

                self.features[feature_index][feature_value] = (feature_value_shows, feature_value_clicks)

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

    def predict(self, test_filename, predictions_filename):
        ctr00 = float(self.clicks_count) / self.shows_count
        predictions_file = open(predictions_filename, mode='w+')

        for line in open(test_filename):
            example_params = parse_vw_line(line)
            example_values = tuple((feature_value for feature_name, feature_value in example_params['features']))

            ctr0 = 1
            for feature_name, feature_value in example_params['features']:
                feature_index = self.features_mapping[feature_name]
                feature_shows, feature_clicks = self.features[feature_index].get(feature_value, (0, 0))
                ctr0 *= (feature_clicks + 2) / (feature_shows + 2/ctr00)

            ctr0 **= 1.0 / len(example_params['features'])

            example_values_shows, example_values_clicks = self.data.get(example_values, (0, 0))
            # print (example_values_clicks + 2) / (example_values_shows + 2/ctr0)
            ctr = (example_values_clicks + 2) / (example_values_shows + 2/ctr0)
            predictions_file.write(str(ctr) + "\n")
