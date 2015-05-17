# -*- coding: utf-8 -*-
import logging

from algorithms.logistic_vw import LogisticVWClassifier
from algorithms.libffm import LibFFMClassifier
from algorithms.dummy import DummyClassifier

from hccf.utils.logs import load_dict_config
from hccf.settings import LOGGING

from hccf.clustering import FeatureClustering
from hccf.utils.helpers import Timer

from hccf.utils.tools import (
    parse_vw_line,
    compose_vw_line,
    parse_libffm_line,
    compose_libffm_line,
)


log = logging.getLogger(__name__)
LOG_FILENAME = 'natural_data_experiments.log'


NUMBER_OF_EXPERIMENTS = 50
NUMBER_OF_PROCESSES = 32
CLUSTERING_LEVELS = 2
DEBUG = True

MODEL_NAME = 'yandex_1000000'
TRAIN_FILE = MODEL_NAME + '-train'
TEST_FILE = MODEL_NAME + '-test'
TRAIN_FILE_CLUSTERING = TRAIN_FILE + '-clusters'
TEST_FILE_CLUSTERING = TEST_FILE + '-clusters'


VW_TREE_FEATURES = ['UserGroupID']
VW_SLICES = ['RegionID']  # 'BannerCTR', 'DeviceType', 'BMCategory1ID'
# LIBFFM_TREE_FEATURES = ['0', '1']


VW_PARAMS = {
    'passes': 100,
    'bit_precision': 28,
    # 'q': ['ab'],
}


def experiments(debug=False, clustering_levels=2, clustering_processes=1):
    with Timer('converted datasets'):
        f_clustering = FeatureClustering(clustering_processes)
        f_clustering.cluster(
            input_logfile=TRAIN_FILE + '.vw',
            tree_features=VW_TREE_FEATURES,
            slice_features=VW_SLICES,
            parser=parse_vw_line,
        )
        f_clustering.convert_log(
            TRAIN_FILE + '.vw',
            TRAIN_FILE_CLUSTERING + '.vw',
            {'mode': 'part_tree', 'levels': clustering_levels},
            parser=parse_vw_line,
            composer=compose_vw_line,
        )
        f_clustering.convert_log(
            TEST_FILE + '.vw',
            TEST_FILE_CLUSTERING + '.vw',
            {'mode': 'part_tree', 'levels': clustering_levels},
            parser=parse_vw_line,
            composer=compose_vw_line,
        )

        # f_clustering = FeatureClustering(clustering_processes)
        # f_clustering.cluster(
        #     input_logfile=TRAIN_FILE + '.libffm',
        #     tree_features=LIBFFM_TREE_FEATURES,
        #     slice_features=LIBFFM_SLICES,
        #     parser=parse_libffm_line,
        # )
        # f_clustering.convert_log(
        #     TRAIN_FILE + '.libffm',
        #     TRAIN_FILE_CLUSTERING + '.libffm',
        #     {'mode': 'part_tree', 'levels': clustering_levels},
        #     parser=parse_libffm_line,
        #     composer=compose_libffm_line,
        # )
        # f_clustering.convert_log(
        #     TEST_FILE + '.libffm',
        #     TEST_FILE_CLUSTERING + '.libffm',
        #     {'mode': 'part_tree', 'levels': clustering_levels},
        #     parser=parse_libffm_line,
        #     composer=compose_libffm_line,
        # )

    results = {}

    # with Timer('LibFFMClassifier'):
    #     ctr_model = LibFFMClassifier(debug=debug)
    #     ctr_model.train(TRAIN_FILE + '.libffm')
    #     train_ll = ctr_model.test(TRAIN_FILE + '.libffm')
    #     test_ll = ctr_model.test(TEST_FILE + '.libffm')
    #     results['LibFFMClassifier'] = {
    #         'train': train_ll,
    #         'test': test_ll,
    #     }
    #
    #     log.info('LibFFMClassifier LL train - %s', train_ll)
    #     log.info('LibFFMClassifier LL test - %s', test_ll)
    #
    # with Timer('LibFFMClassifierClustering'):
    #     ctr_model = LibFFMClassifier(debug=debug)
    #     ctr_model.train(TRAIN_FILE_CLUSTERING + '.libffm')
    #     train_ll = ctr_model.test(TRAIN_FILE_CLUSTERING + '.libffm')
    #     test_ll = ctr_model.test(TEST_FILE_CLUSTERING + '.libffm')
    #     results['LibFFMClassifierClustering'] = {
    #         'train': train_ll,
    #         'test': test_ll,
    #     }
    #
    #     log.info('LibFFMClassifierClustering LL train - %s', train_ll)
    #     log.info('LibFFMClassifierClustering LL test - %s', test_ll)

    with Timer('DummyClassifier'):
        ctr_model = DummyClassifier()
        ctr_model.train(TRAIN_FILE + '.vw')
        train_ll = ctr_model.test(TRAIN_FILE + '.vw')
        test_ll = ctr_model.test(TEST_FILE + '.vw')
        results['DummyClassifier'] = {
            'train': train_ll,
            'test': test_ll,
        }

        log.info('DummyClassifier LL0 - %s', ctr_model.loglikelihood0())
        log.info('DummyClassifier LL train - %s', train_ll)
        log.info('DummyClassifier LL test - %s', test_ll)

    with Timer('LogisticVWClassifier'):
        ctr_model = LogisticVWClassifier(
            debug=debug,
            **VW_PARAMS
        )
        ctr_model.train(TRAIN_FILE + '.vw')
        train_ll = ctr_model.test(TRAIN_FILE + '.vw')
        test_ll = ctr_model.test(TEST_FILE + '.vw')
        results['LogisticVWClassifier'] = {
            'train': train_ll,
            'test': test_ll,
        }

        log.info('LogisticVWClassifier LL train - %s', train_ll)
        log.info('LogisticVWClassifier LL test - %s', test_ll)

    with Timer('LogisticVWClassifierClustering'):
        ctr_model = LogisticVWClassifier(
            debug=debug,
            **VW_PARAMS
        )
        ctr_model.train(TRAIN_FILE_CLUSTERING + '.vw')
        train_ll = ctr_model.test(TRAIN_FILE_CLUSTERING + '.vw')
        test_ll = ctr_model.test(TEST_FILE_CLUSTERING + '.vw')
        results['LogisticVWClassifierClustering'] = {
            'train': train_ll,
            'test': test_ll,
        }

        log.info('LogisticVWClassifierClustering LL train - %s', train_ll)
        log.info('LogisticVWClassifierClustering LL test - %s', test_ll)

    with Timer('FTRLVWClassifier'):
        ctr_model = LogisticVWClassifier(
            debug=debug,
            ftrl=True,
            ftrl_alpha=0.1,
            ftrl_beta=0.0,
            **VW_PARAMS
        )
        ctr_model.train(TRAIN_FILE + '.vw')
        train_ll = ctr_model.test(TRAIN_FILE + '.vw')
        test_ll = ctr_model.test(TEST_FILE + '.vw')
        results['FTRLVWClassifier'] = {
            'train': train_ll,
            'test': test_ll,
        }

        log.info('FTRLVWClassifier FTRL LL train - %s', train_ll)
        log.info('FTRLVWClassifier FTRL LL test - %s', test_ll)

    with Timer('FTRLVWClassifierClustering'):
        ctr_model = LogisticVWClassifier(
            debug=debug,
            ftrl=True,
            ftrl_alpha=0.1,
            ftrl_beta=0.0,
            **VW_PARAMS
        )
        ctr_model.train(TRAIN_FILE_CLUSTERING + '.vw')
        train_ll = ctr_model.test(TRAIN_FILE_CLUSTERING + '.vw')
        test_ll = ctr_model.test(TEST_FILE_CLUSTERING + '.vw')
        results['FTRLVWClassifierClustering'] = {
            'train': train_ll,
            'test': test_ll,
        }

        log.info('FTRLVWClassifierClustering LL train - %s', train_ll)
        log.info('FTRLVWClassifierClustering LL test - %s', test_ll)

    return results


def main():
    experiments(
        debug=DEBUG,
        clustering_levels=CLUSTERING_LEVELS,
        clustering_processes=NUMBER_OF_PROCESSES,
    )


if __name__ == '__main__':
    logging_level = logging.DEBUG if DEBUG else logging.INFO
    load_dict_config(LOGGING, logging_level, filename=LOG_FILENAME)
    main()
