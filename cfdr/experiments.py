# -*- coding: utf-8 -*-
import logging

from cfdr.ctr_model import CTRModel
from cfdr.models.logistic_vw import LogisticVWClassifier
from cfdr.models.historic import HistoricClassifier
from cfdr.clustering import FeatureClustering
from cfdr.utils.helpers import Timer
from cfdr import settings


log = logging.getLogger(__name__)


FEATURES_CONFIG = {
    'a': {
        'count': 64,
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


def ctr_historic(model='clicklog', from_cache=False, train_dataset_length=100000, test_dataset_length=100000):
    TRAIN_FILE = model + '.train.vw'
    TEST_FILE = model + '.test.vw'

    if from_cache:
        real_ctr_model = CTRModel.load(model + '.dat')
    else:
        with Timer('init real model'):
            real_ctr_model = CTRModel(FEATURES_CONFIG, free_coef=-1, lam=20)
            real_ctr_model.init()

        with Timer('generate clicklog'):
            real_ctr_model.generate_log(
                filename=model,
                format='vw',
                train_length=train_dataset_length,
                test_length=test_dataset_length,
            )
            real_ctr_model.save(model + '.dat')

    with Timer('calculate likelihood'):
        ll = real_ctr_model.loglikelihood()
        ll0 = real_ctr_model.loglikelihood0()
        likelihood_ratio = real_ctr_model.likelihood_ratio()
        log.info('loglikelihood = %s', ll)
        log.info('loglikelihood0 = %s', ll0)
        log.info('likelihood_ratio = %s', likelihood_ratio)

    model = HistoricClassifier()
    model.train(TRAIN_FILE)
    train_ll = model.test(TRAIN_FILE)
    test_ll = model.test(TEST_FILE)

    log.info('LL train - %s', train_ll)
    log.info('LL test - %s', test_ll)
    log.info('LLr train - %s', (ll0 - train_ll))
    log.info('LLr test - %s', (ll0 - test_ll))


def ctr_logistic_vw(model='clicklog', from_cache=False, debug=False, train_dataset_length=100000, test_dataset_length=100000, index=0, with_clustering=True):
    TRAIN_FILE = model + '.train.vw'
    TEST_FILE = model + '.test.vw'

    TRAIN_FILE_CLUSTERING = model + '-clusters.train.vw'
    TEST_FILE_CLUSTERING = model + '-clusters.test.vw'

    if from_cache:
        real_ctr_model = CTRModel.load(model + '.dat')
    else:
        with Timer('init real model'):
            real_ctr_model = CTRModel(FEATURES_CONFIG, free_coef=-1, lam=20)
            real_ctr_model.init()

        with Timer('generate clicklog'):
            real_ctr_model.generate_log(
                filename=model,
                format='vw',
                train_length=train_dataset_length,
                test_length=test_dataset_length,
            )
            real_ctr_model.save(model + '.dat')

    if with_clustering:
        f_clustering = FeatureClustering()
        with Timer('clustering'):
            f_clustering.cluster(TRAIN_FILE, TREE_FEATURES)
        with Timer('converted train set'):
            f_clustering.convert_log(model + '.train.vw', TRAIN_FILE_CLUSTERING + "2", {'mode': 'part_tree', 'levels': 2})
            f_clustering.convert_log(model + '.train.vw', TRAIN_FILE_CLUSTERING + "3", {'mode': 'part_tree', 'levels': 3})
            f_clustering.convert_log(model + '.train.vw', TRAIN_FILE_CLUSTERING + "4", {'mode': 'part_tree', 'levels': 4})
        with Timer('converted test set'):
            f_clustering.convert_log(model + '.test.vw', TEST_FILE_CLUSTERING + "2", {'mode': 'part_tree', 'levels': 2})
            f_clustering.convert_log(model + '.test.vw', TEST_FILE_CLUSTERING + "3", {'mode': 'part_tree', 'levels': 3})
            f_clustering.convert_log(model + '.test.vw', TEST_FILE_CLUSTERING + "4", {'mode': 'part_tree', 'levels': 4})

    with Timer('calculate likelihood'):
        ll = real_ctr_model.loglikelihood()
        ll0 = real_ctr_model.loglikelihood0()
        likelihood_ratio = real_ctr_model.likelihood_ratio()
        log.info('loglikelihood = %s', ll)
        log.info('loglikelihood0 = %s', ll0)
        log.info('likelihood_ratio = %s', likelihood_ratio)

    results = []

    ctr_prediction = LogisticVWClassifier(debug=debug, passes=200, bit_precision=13)
    baseline_model_log_loss = ctr_prediction.run(TRAIN_FILE, TEST_FILE)
    log.info('usual model log loss = %s', baseline_model_log_loss)
    results.append((train_dataset_length, index, ll, baseline_model_log_loss))

    if with_clustering:
        ctr_prediction = LogisticVWClassifier(debug=debug, passes=200, bit_precision=13)
        clustering_model_log_loss = ctr_prediction.run(TRAIN_FILE_CLUSTERING + "2", TEST_FILE_CLUSTERING + "2")
        log.info('usual model log loss = %s', clustering_model_log_loss)
        results.append((train_dataset_length, index, ll, clustering_model_log_loss))
        log.info('LLr baseline - clistering = %s',
                 (baseline_model_log_loss[0] - clustering_model_log_loss[0],
                  baseline_model_log_loss[1] - clustering_model_log_loss[1]))

        # clustering_model_log_loss = ctr_prediction.run(TRAIN_FILE_CLUSTERING + "3", TEST_FILE_CLUSTERING + "3")
        # log.info('usual model log loss = %s', clustering_model_log_loss)
        # results.append((train_dataset_length, index, ll, clustering_model_log_loss))
        # log.info('LLr baseline - clistering = %s',
        #          (baseline_model_log_loss[0] - clustering_model_log_loss[0],
        #           baseline_model_log_loss[1] - clustering_model_log_loss[1]))
        #
        # clustering_model_log_loss = ctr_prediction.run(TRAIN_FILE_CLUSTERING + "4", TEST_FILE_CLUSTERING + "4")
        # log.info('usual model log loss = %s', clustering_model_log_loss)
        # results.append((train_dataset_length, index, ll, clustering_model_log_loss))
        # log.info('LLr baseline - clistering = %s',
        #          (baseline_model_log_loss[0] - clustering_model_log_loss[0],
        #           baseline_model_log_loss[1] - clustering_model_log_loss[1]))

    return results


if __name__ == '__main__':
    # results = grid_prediction(debug=True)
    # results = ctr_prediction(model='clicklog2-output', dataset_length=100000, debug=True, index=1)

    ctr_historic(
        model='clicklog',
        from_cache=False,
        train_dataset_length=50000,
        test_dataset_length=50000,
    )

    results = ctr_logistic_vw(
        model='clicklog',
        from_cache=True,
        debug=False,
        index=1,
        train_dataset_length=50000,
        test_dataset_length=50000,
        with_clustering=True,
    )
    print results

