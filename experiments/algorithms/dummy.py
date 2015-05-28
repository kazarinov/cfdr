# -*- coding: utf-8 -*-
import logging

from hccf.utils.tools import parse_vw_line
from hccf.utils.mathematics import loglikelihood

from .base import BaseClassifier


log = logging.getLogger(__name__)


class DummyClassifier(BaseClassifier):
    def __init__(self):
        self.shows_count = 0
        self.clicks_count = 0

    def train(self, train_filename, parser=None):
        if parser is None:
            parser = parse_vw_line

        for line in open(train_filename):
            example_params = parser(line)
            self.shows_count += 1
            if example_params['label'] > 0:
                self.clicks_count += 1

    def ctr(self):
        return self.clicks_count / float(self.shows_count)

    def loglikelihood0(self):
        return loglikelihood(self.shows_count, self.clicks_count) / self.shows_count

    def predict(self, test_filename, predictions_filename):
        ctr = self.ctr()
        predictions_file = open(predictions_filename, mode='w+')

        for line in open(test_filename):
            predictions_file.write(str(ctr) + "\n")

        predictions_file.close()
