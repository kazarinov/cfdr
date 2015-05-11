# -*- coding: utf-8 -*-
import logging

from hccf.experiments.ctr_model import CTRModel
from hccf.experiments.models.logistic_vw import LogisticVWClassifier
from hccf.experiments.models.historic import HistoricClassifier
from hccf.experiments.models.libffm import LibFFMClassifier
from hccf.clustering import FeatureClustering
from hccf.utils.helpers import Timer
from hccf.utils.logs import load_dict_config
from hccf.utils.tools import (
    parse_vw_line,
    parse_libffm_line,
    compose_libffm_line,
    compose_vw_line,
)
from hccf.settings import LOGGING


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

VW_TREE_FEATURES = ['a', 'b']
LIBFFM_TREE_FEATURES = ['0', '1']


def ctr_historic(model='clicklog', from_cache=False, train_dataset_length=100000, test_dataset_length=100000):
    TRAIN_FILE = model + '.train.vw'
    TEST_FILE = model + '.test.vw'

    if from_cache:
        real_ctr_model = CTRModel.load(model + '.dat')
    else:
        with Timer('init real model'):
            real_ctr_model = CTRModel(FEATURES_CONFIG, free_coef=-1, lam=200)
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


def ctr_libffm(model='clicklog', from_cache=False, debug=False, train_dataset_length=100000, test_dataset_length=100000):
    TRAIN_FILE = model + '.train'
    TEST_FILE = model + '.test'
    TRAIN_FILE_CLUSTERING = model + '-clusters.train'
    TEST_FILE_CLUSTERING = model + '-clusters.test'

    if from_cache:
        real_ctr_model = CTRModel.load(model + '.dat')
    else:
        with Timer('init real model'):
            real_ctr_model = CTRModel(FEATURES_CONFIG, free_coef=-1, lam=200)
            real_ctr_model.init()

        with Timer('generate clicklog'):
            real_ctr_model.generate_log(
                filename=model,
                format='libffm',
                train_length=train_dataset_length,
                test_length=test_dataset_length,
            )
            real_ctr_model.generate_log(
                filename=model,
                format='vw',
                train_length=train_dataset_length,
                test_length=test_dataset_length,
            )
            real_ctr_model.save(model + '.dat')

        with Timer('converted datasets'):
            f_clustering = FeatureClustering()
            f_clustering.cluster(TRAIN_FILE + '.vw', VW_TREE_FEATURES, parser=parse_vw_line)
            f_clustering.convert_log(
                TRAIN_FILE + '.vw',
                TRAIN_FILE_CLUSTERING + '.vw',
                {'mode': 'part_tree', 'levels': 2},
                parser=parse_vw_line,
                composer=compose_vw_line,
            )
            f_clustering.convert_log(
                TEST_FILE + '.vw',
                TEST_FILE_CLUSTERING + '.vw',
                {'mode': 'part_tree', 'levels': 2},
                parser=parse_vw_line,
                composer=compose_vw_line,
            )

            f_clustering = FeatureClustering()
            f_clustering.cluster(TRAIN_FILE + '.libffm', LIBFFM_TREE_FEATURES, parser=parse_libffm_line)
            f_clustering.convert_log(
                TRAIN_FILE + '.libffm',
                TRAIN_FILE_CLUSTERING + '.libffm',
                {'mode': 'part_tree', 'levels': 2},
                parser=parse_libffm_line,
                composer=compose_libffm_line,
            )
            f_clustering.convert_log(
                TEST_FILE + '.libffm',
                TEST_FILE_CLUSTERING + '.libffm',
                {'mode': 'part_tree', 'levels': 2},
                parser=parse_libffm_line,
                composer=compose_libffm_line,
            )

    results = {}

    with Timer('calculate likelihood'):
        ll = real_ctr_model.loglikelihood()
        ll0 = real_ctr_model.loglikelihood0()
        likelihood_ratio = real_ctr_model.likelihood_ratio()
        results['True'] = (ll, ll0)

        log.info('loglikelihood = %s', ll)
        log.info('loglikelihood0 = %s', ll0)
        log.info('likelihood_ratio = %s', likelihood_ratio)

    with Timer('LibFFMClassifier'):
        ctr_model = LibFFMClassifier(debug=debug)
        ctr_model.train(TRAIN_FILE + '.libffm')
        train_ll = ctr_model.test(TRAIN_FILE + '.libffm')
        test_ll = ctr_model.test(TEST_FILE + '.libffm')
        results['LibFFMClassifier'] = ((train_ll, (ll0 - train_ll)), (test_ll, (ll0 - test_ll)))

        log.info('LibFFMClassifier LL train - %s', train_ll)
        log.info('LibFFMClassifier LL test - %s', test_ll)
        log.info('LibFFMClassifier LLr train - %s', (ll0 - train_ll))
        log.info('LibFFMClassifier LLr test - %s', (ll0 - test_ll))

    with Timer('LibFFMClassifier with clustering'):
        ctr_model = LibFFMClassifier(debug=debug)
        ctr_model.train(TRAIN_FILE_CLUSTERING + '.libffm')
        train_ll = ctr_model.test(TRAIN_FILE_CLUSTERING + '.libffm')
        test_ll = ctr_model.test(TEST_FILE_CLUSTERING + '.libffm')
        results['LibFFMClassifier with clustering'] = ((train_ll, (ll0 - train_ll)), (test_ll, (ll0 - test_ll)))

        log.info('LibFFMClassifier with clustering LL train - %s', train_ll)
        log.info('LibFFMClassifier with clustering LL test - %s', test_ll)
        log.info('LibFFMClassifier with clustering LLr train - %s', (ll0 - train_ll))
        log.info('LibFFMClassifier with clustering LLr test - %s', (ll0 - test_ll))

    # with Timer('HistoricClassifier'):
    #     ctr_model = HistoricClassifier()
    #     ctr_model.train(TRAIN_FILE + '.vw')
    #     train_ll = ctr_model.test(TRAIN_FILE + '.vw')
    #     test_ll = ctr_model.test(TEST_FILE + '.vw')
    #
    #     log.info('HistoricClassifier LL train - %s', train_ll)
    #     log.info('HistoricClassifier LL test - %s', test_ll)
    #     log.info('HistoricClassifier LLr train - %s', (ll0 - train_ll))
    #     log.info('HistoricClassifier LLr test - %s', (ll0 - test_ll))

    with Timer('LogisticVWClassifier'):
        ctr_model = LogisticVWClassifier(debug=debug, passes=100, bit_precision=13)
        ctr_model.train(TRAIN_FILE + '.vw')
        train_ll = ctr_model.test(TRAIN_FILE + '.vw')
        test_ll = ctr_model.test(TEST_FILE + '.vw')
        results['LogisticVWClassifier'] = ((train_ll, (ll0 - train_ll)), (test_ll, (ll0 - test_ll)))

        log.info('LogisticVWClassifier LL train - %s', train_ll)
        log.info('LogisticVWClassifier LL test - %s', test_ll)
        log.info('LogisticVWClassifier LLr train - %s', (ll0 - train_ll))
        log.info('LogisticVWClassifier LLr test - %s', (ll0 - test_ll))

    with Timer('LogisticVWClassifier with clustering'):
        ctr_model = LogisticVWClassifier(debug=debug, passes=100, bit_precision=13)
        ctr_model.train(TRAIN_FILE_CLUSTERING + '.vw')
        train_ll = ctr_model.test(TRAIN_FILE_CLUSTERING + '.vw')
        test_ll = ctr_model.test(TEST_FILE_CLUSTERING + '.vw')
        results['LogisticVWClassifier with clustering'] = ((train_ll, (ll0 - train_ll)), (test_ll, (ll0 - test_ll)))

        log.info('LogisticVWClassifier with clustering LL train - %s', train_ll)
        log.info('LogisticVWClassifier with clustering LL test - %s', test_ll)
        log.info('LogisticVWClassifier with clusteringLLr train - %s', (ll0 - train_ll))
        log.info('LogisticVWClassifier with clusteringLLr test - %s', (ll0 - test_ll))

    with Timer('LogisticVWClassifier FTRL'):
        ctr_model = LogisticVWClassifier(
            debug=debug,
            passes=100,
            bit_precision=13,

            ftrl=True,
            ftrl_alpha=0.1,
            ftrl_beta=0.0,
        )
        ctr_model.train(TRAIN_FILE + '.vw')
        train_ll = ctr_model.test(TRAIN_FILE + '.vw')
        test_ll = ctr_model.test(TEST_FILE + '.vw')
        results['LogisticVWClassifier'] = ((train_ll, (ll0 - train_ll)), (test_ll, (ll0 - test_ll)))

        log.info('LogisticVWClassifier FTRL LL train - %s', train_ll)
        log.info('LogisticVWClassifier FTRL LL test - %s', test_ll)
        log.info('LogisticVWClassifier FTRL LLr train - %s', (ll0 - train_ll))
        log.info('LogisticVWClassifier FTRL LLr test - %s', (ll0 - test_ll))

    with Timer('LogisticVWClassifier FTRL with clustering'):
        ctr_model = LogisticVWClassifier(
            debug=debug,
            passes=100,
            bit_precision=13,

            ftrl=True,
            ftrl_alpha=0.1,
            ftrl_beta=0.0,
        )
        ctr_model.train(TRAIN_FILE_CLUSTERING + '.vw')
        train_ll = ctr_model.test(TRAIN_FILE_CLUSTERING + '.vw')
        test_ll = ctr_model.test(TEST_FILE_CLUSTERING + '.vw')
        results['LogisticVWClassifier FTRL with clustering'] = ((train_ll, (ll0 - train_ll)), (test_ll, (ll0 - test_ll)))

        log.info('LogisticVWClassifier FTRL with clustering LL train - %s', train_ll)
        log.info('LogisticVWClassifier FTRL with clustering LL test - %s', test_ll)
        log.info('LogisticVWClassifier FTRL with clusteringLLr train - %s', (ll0 - train_ll))
        log.info('LogisticVWClassifier FTRL with clusteringLLr test - %s', (ll0 - test_ll))


def ctr_logistic_vw(model='clicklog', from_cache=False, debug=False, train_dataset_length=100000, test_dataset_length=100000, index=0, with_clustering=True):
    TRAIN_FILE = model + '.train.vw'
    TEST_FILE = model + '.test.vw'

    TRAIN_FILE_CLUSTERING = model + '-clusters.train.vw'
    TEST_FILE_CLUSTERING = model + '-clusters.test.vw'

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
                train_length=train_dataset_length,
                test_length=test_dataset_length,
            )
            real_ctr_model.save(model + '.dat')

    if with_clustering:
        f_clustering = FeatureClustering()
        with Timer('clustering'):
            f_clustering.cluster(TRAIN_FILE, VW_TREE_FEATURES)
        with Timer('converted train set'):
            f_clustering.convert_log(model + '.train.vw', TRAIN_FILE_CLUSTERING + "2", {'mode': 'part_tree', 'levels': 2})
            # f_clustering.convert_log(model + '.train.vw', TRAIN_FILE_CLUSTERING + "3", {'mode': 'part_tree', 'levels': 3})
            # f_clustering.convert_log(model + '.train.vw', TRAIN_FILE_CLUSTERING + "4", {'mode': 'part_tree', 'levels': 4})
        with Timer('converted test set'):
            f_clustering.convert_log(model + '.test.vw', TEST_FILE_CLUSTERING + "2", {'mode': 'part_tree', 'levels': 2})
            # f_clustering.convert_log(model + '.test.vw', TEST_FILE_CLUSTERING + "3", {'mode': 'part_tree', 'levels': 3})
            # f_clustering.convert_log(model + '.test.vw', TEST_FILE_CLUSTERING + "4", {'mode': 'part_tree', 'levels': 4})

    with Timer('calculate likelihood'):
        ll = real_ctr_model.loglikelihood()
        ll0 = real_ctr_model.loglikelihood0()
        likelihood_ratio = real_ctr_model.likelihood_ratio()
        log.info('loglikelihood = %s', ll)
        log.info('loglikelihood0 = %s', ll0)
        log.info('likelihood_ratio = %s', likelihood_ratio)

    results = []

    ctr_prediction = LogisticVWClassifier(
        debug=debug,
        passes=100,
        bit_precision=13,
    )
    baseline_model_log_loss = ctr_prediction.run(TRAIN_FILE, TEST_FILE)
    log.info('LogisticVWClassifier LL = %s', baseline_model_log_loss)
    results.append((train_dataset_length, index, ll, baseline_model_log_loss))

    ctr_prediction = LogisticVWClassifier(
        debug=debug,
        passes=100,
        bit_precision=13,

        ftrl=True,
        ftrl_alpha=0.1,
        ftrl_beta=0.0,
    )
    ftrl_model_log_loss = ctr_prediction.run(TRAIN_FILE, TEST_FILE)
    log.info('FTRL LogisticVWClassifier LL = %s', baseline_model_log_loss)
    results.append((train_dataset_length, index, ll, ftrl_model_log_loss))
    log.info('LLr baseline - clistering = %s',
                 (baseline_model_log_loss[0] - ftrl_model_log_loss[0],
                  baseline_model_log_loss[1] - ftrl_model_log_loss[1]))

    if with_clustering:
        ctr_prediction = LogisticVWClassifier(
            debug=debug,
            passes=100,
            bit_precision=13,
        )
        clustering_model_log_loss = ctr_prediction.run(TRAIN_FILE_CLUSTERING + "2", TEST_FILE_CLUSTERING + "2")
        log.info('LogisticVWClassifier with clustering = %s', clustering_model_log_loss)
        results.append((train_dataset_length, index, ll, clustering_model_log_loss))
        log.info('LLr baseline - clistering = %s',
                 (baseline_model_log_loss[0] - clustering_model_log_loss[0],
                  baseline_model_log_loss[1] - clustering_model_log_loss[1]))

        ctr_prediction = LogisticVWClassifier(
            debug=debug,
            passes=100,
            bit_precision=13,

            ftrl=True,
            ftrl_alpha=0.1,
            ftrl_beta=0.0,
        )
        clustering_model_log_loss = ctr_prediction.run(TRAIN_FILE_CLUSTERING + "2", TEST_FILE_CLUSTERING + "2")
        log.info('FTRL LogisticVWClassifier with clustering = %s', clustering_model_log_loss)
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
    load_dict_config(LOGGING)
    # results = grid_prediction(debug=True)
    # results = ctr_prediction(model='clicklog2-output', dataset_length=100000, debug=True, index=1)

    # ctr_historic(
    #     model='clicklog',
    #     from_cache=False,
    #     train_dataset_length=50000,
    #     test_dataset_length=50000,
    # )

    # results = ctr_logistic_vw(
    #     model='clicklog',
    #     from_cache=False,
    #     debug=True,
    #     index=1,
    #     train_dataset_length=500000,
    #     test_dataset_length=500000,
    #     with_clustering=True,
    # )
    # print results

    ctr_libffm(
        model='clicklog',
        from_cache=False,
        debug=True,
        train_dataset_length=500000,
        test_dataset_length=500000,
    )
