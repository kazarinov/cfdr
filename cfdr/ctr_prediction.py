# -*- coding: utf-8 -*-
import logging

import numpy as np
import pandas as pd
from fabric.api import local


from cfdr.ctr_model import CTRModel
from cfdr.utils.helpers import (
    Timer,
    silentremove,
    sigmoid,
    log_loss,
)
from cfdr.utils.vw import make_vw_command
from cfdr.clustering import FeatureClustering
from cfdr import settings


log = logging.getLogger(__name__)


class CTRPrediction(object):
    scoring = 'log_loss'
    classes = [-1, 1]

    def __init__(self, real_ctr_model, vw_params, debug=False):
        self.real_ctr_model = real_ctr_model
        self.vw_params = vw_params
        self.debug = debug
        avarage_ctr = np.sum(self.real_ctr_model.clicks) / np.sum(self.real_ctr_model.shows)

    def preprocess_data(self, filename):
        pass

    def run(self, train_filename, test_filename):
        model_filename = train_filename + '.model'
        predictions_filename = test_filename + '.pred'

        with Timer('preprocess train data'):
            self.preprocess_data(train_filename)

        # train model
        train_params = self.vw_params.copy()
        train_params.update({
            'final_regressor': model_filename,
            'data': train_filename,
            'predictions': False,
            'quiet': False,
        })
        with Timer('train model'):
            train_vw = make_vw_command(**train_params)
            local(train_vw)

        # test model
        test_params = {
            'testonly': True,
            'data': test_filename,
            'initial_regressor': model_filename,
            'predictions': predictions_filename,
            'quiet': True,
        }

        for param in ['hash', 'bit_precision', 'q']:
            if self.vw_params.get(param):
                test_params[param] = self.vw_params[param]

        with Timer('predict test results'):
            test_vw = make_vw_command(**test_params)
            local(test_vw)


        test_data = pd.read_table(test_filename, sep='|', header=None)
        test_data[0] = (test_data[0] + 1) / 2
        actual = np.array(test_data[0])

        predictions_data = pd.read_table(predictions_filename, sep='|', header=None)
        predictions_data[0] = predictions_data[0].map(lambda x: sigmoid(x))
        predictions = np.array(predictions_data[0])

        ll = log_loss(actual, predictions)

        # cleanup
        if not self.debug:
            silentremove(model_filename)
            silentremove(predictions_filename)

        return ll


FEATURES_CONFIG = {
    'a': {
        'count': 128,
        'loc': 0.0,
        'scale': 0.5,
        'type': 'tree',
    },
    'b': {
        'count': 100,
        'loc': 0.0,
        'scale': 0.5,
        'type': 'tree',
    },
    'axb': {
        'loc': 0.0,
        'scale': 0.8,
        'parts': ['a', 'b'],
    }
}

TREE_FEATURES = ['a']


def ctr_prediction(model='clicklog', from_cache=False, debug=False, dataset_length=100000, index=0, with_clustering=True):
    TRAIN_FILE = model + '.train.vw'
    TEST_FILE = model + '.test.vw'

    TRAIN_FILE_CLUSTERING = model + '-clusters.train.vw'
    TEST_FILE_CLUSTERING = model + '-clusters.test.vw'

    vw_model = {
        'kill_cache': True,
        'cache_file': TRAIN_FILE + '.cache',
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

    if from_cache:
        real_ctr_model = CTRModel.load(model + '.dat')
    else:
        with Timer('init real model'):
            real_ctr_model = CTRModel(FEATURES_CONFIG, free_coef=-1, lam=100)
            real_ctr_model.init()

        with Timer('generate clicklog'):
            real_ctr_model.generate_log(
                filename=model,
                format='vw',
                train_length=dataset_length,
                test_length=dataset_length
            )
            real_ctr_model.save(model + '.dat')

    if with_clustering:
        f_clustering = FeatureClustering()
        with Timer('clustering'):
            f_clustering.cluster(TRAIN_FILE, TREE_FEATURES)
        with Timer('converted train set'):
            f_clustering.convert_log(model + '.train.vw', model + '-output.train.vw', {'mode': 'full_tree'})
        with Timer('converted test set'):
            f_clustering.convert_log(model + '.test.vw', model + '-output.test.vw', {'mode': 'full_tree'})

    with Timer('calculate likelihood'):
        ll = real_ctr_model.loglikelihood()
        ll0 = real_ctr_model.loglikelihood0()
        likelihood_ratio = real_ctr_model.likelihood_ratio()
        log.info('loglikelihood = %s', ll)
        log.info('loglikelihood0 = %s', ll0)
        log.info('likelihood_ratio = %s', likelihood_ratio)

    results = []

    ctr_prediction = CTRPrediction(real_ctr_model, vw_model, debug=debug)
    model_log_loss = ctr_prediction.run(TRAIN_FILE, TEST_FILE)
    log.info('usual model log loss = %s', model_log_loss)
    results.append((dataset_length, index, ll, model_log_loss))

    if with_clustering:
        ctr_prediction = CTRPrediction(real_ctr_model, vw_model, debug=debug)
        model_log_loss = ctr_prediction.run(TRAIN_FILE_CLUSTERING, TEST_FILE_CLUSTERING)
        log.info('usual model log loss = %s', model_log_loss)
        results.append((dataset_length, index, ll, model_log_loss))

    if not debug:
        silentremove(TRAIN_FILE)
        silentremove(TEST_FILE)

    return results


def grid_prediction(debug=True):
    results = []
    for dataset_length in [100000, 200000, 500000, 1000000]:
        for i in xrange(10):
            results += ctr_prediction(dataset_length=dataset_length, debug=debug, index=i)
    return results


if __name__ == '__main__':
    # results = grid_prediction(debug=True)
    # results = ctr_prediction(model='clicklog2-output', dataset_length=100000, debug=True, index=1)
    results = ctr_prediction(model='clicklog', from_cache=False, debug=True, index=1, dataset_length=500000, with_clustering=True)
    print results
