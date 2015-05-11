# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from hccf.utils.mathematics import log_loss


class BaseClassifier(object):

    def _get_predictions(self, predictions_filename):
        predictions_data = pd.read_table(predictions_filename, sep='|', header=None)
        predictions = np.array(predictions_data[0])
        return predictions

    def _get_actual_labels(self, filename):
        actual_labels = pd.read_table(filename, sep='|', header=None)
        actual_labels[0] = (actual_labels[0] + 1) / 2
        actual = np.array(actual_labels[0])
        return actual

    def test(self, test_filename):
        predictions_filename = test_filename + '.pred'
        self.predict(test_filename, predictions_filename)

        actual = self._get_actual_labels(test_filename)
        predictions = self._get_predictions(predictions_filename)
        ll = log_loss(actual, predictions)
        return ll

    def run(self, train_filename, test_filename):
        self.train(train_filename)

        train_ll = self.test(train_filename)
        test_ll = self.test(test_filename)

        return train_ll, test_ll

    def preprocess_data(self, filename):
        pass

    def train(self, train_filename):
        raise NotImplementedError

    def predict(self, test_filename, predictions_filename):
        raise NotImplementedError
