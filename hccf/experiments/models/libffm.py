# -*- coding: utf-8 -*-
import logging

import numpy as np
from fabric.api import local

from hccf.utils.tools import make_command, parse_libffm_line
from hccf.utils.helpers import Timer
from hccf.experiments.models.base import BaseClassifier


log = logging.getLogger(__name__)


class LibFFMClassifier(BaseClassifier):

    def __init__(self, model_filename=None, debug=False, **kwargs):
        self.model_params = {
            'l': 1e-8,
            'k': 32,
            't': 100,
            'r': 0.01,
            's': 4,
            'norm': True,
            'quiet': not debug,
        }
        self.model_params.update(kwargs)
        self.debug = debug

        self.model_filename = model_filename

    def _get_actual_labels(self, filename):
        actual_labels = []
        for line in open(filename):
            example = parse_libffm_line(line)
            actual_labels.append(int(example['label']))
        return np.array(actual_labels)

    def train(self, train_filename):
        if self.model_filename is None:
            self.model_filename = train_filename + '.model'

        with Timer('preprocess train data'):
            self.preprocess_data(train_filename)

        with Timer('train model'):
            train_command = make_command('libffm/ffm-train', train_filename, self.model_filename, **self.model_params)
            local(train_command)

    def predict(self, test_filename, predictions_filename):
        with Timer('preprocess test data'):
            self.preprocess_data(test_filename)

        # test model on test data
        with Timer('predict results on test data'):
            test_command = make_command('libffm/ffm-predict', test_filename, self.model_filename, predictions_filename)
            local(test_command)
