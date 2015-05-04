# -*- coding: utf-8 -*-
import logging

import numpy as np
import pandas as pd
from fabric.api import local

from cfdr.utils.tools import make_vw_command
from cfdr.utils.helpers import Timer
from cfdr.utils.mathematics import sigmoid

from .base import BaseClassifier


log = logging.getLogger(__name__)


class LogisticVWClassifier(BaseClassifier):

    def __init__(self, model_filename=None, debug=False, **kwargs):
        self.vw_params = {
            'kill_cache': True,
            'cache_file': 'ctr_prediction.cache',
            'loss_function': 'logistic',
            'learning_rate': 0.01,
            'power_t': 0.5,
            'passes': 100,
            'bit_precision': 20,
            'hash': 'all',
            # 'l1': 1e-8,
            'l2': 1e-8,
            'q': ['ab'],
        }
        self.vw_params.update(kwargs)
        self.debug = debug

        self.model_filename = model_filename

    def _get_predictions(self, predictions_filename):
        predictions_data = pd.read_table(predictions_filename, sep='|', header=None)
        predictions_data[0] = predictions_data[0].map(lambda x: sigmoid(x))
        predictions = np.array(predictions_data[0])
        return predictions

    def train(self, train_filename):
        if self.model_filename is None:
            self.model_filename = train_filename + '.model'

        with Timer('preprocess train data'):
            self.preprocess_data(train_filename)

        # train model
        train_params = self.vw_params.copy()
        train_params.update({
            'final_regressor': self.model_filename,
            'data': train_filename,
            'predictions': False,
            'quiet': not self.debug,
        })
        with Timer('train model'):
            train_vw = make_vw_command(**train_params)
            local(train_vw)

    def predict(self, test_filename, predictions_filename):
        with Timer('preprocess test data'):
            self.preprocess_data(test_filename)

        test_params = {
            'testonly': True,
            'data': None,
            'initial_regressor': self.model_filename,
            'predictions': None,
            'quiet': not self.debug,
        }

        for param in ['hash', 'bit_precision', 'q']:
            if self.vw_params.get(param):
                test_params[param] = self.vw_params[param]

        # test model on test data
        with Timer('predict results on test data'):
            test_params['data'] = test_filename
            test_params['predictions'] = predictions_filename
            test_vw = make_vw_command(**test_params)
            local(test_vw)
