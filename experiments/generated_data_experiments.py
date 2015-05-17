# -*- coding: utf-8 -*-
import logging

import numpy as np

from ctr_model import CTRModel
from algorithms.logistic_vw import LogisticVWClassifier
from algorithms.libffm import LibFFMClassifier

from hccf.clustering import FeatureClustering
from hccf.utils.helpers import Timer
from hccf.utils.logs import load_dict_config
from hccf.utils.tools import (
    parse_vw_line,
    compose_vw_line,
    parse_libffm_line,
    compose_libffm_line,
)
from hccf.settings import LOGGING


log = logging.getLogger(__name__)
LOG_FILENAME = 'generated_data_experiments.log'

TRAIN_DATASET_LENGTH = 500000
TEST_DATASET_LENGTH = 500000
CLUSTERING_LEVELS = 2

NUMBER_OF_EXPERIMENTS = 50
NUMBER_OF_PROCESSES = 32

DEBUG = True
FROM_CACHE = False

MODEL_NAME = 'generated_clicklog'
TRAIN_FILE = MODEL_NAME + '.train'
TEST_FILE = MODEL_NAME + '.test'
TRAIN_FILE_CLUSTERING = MODEL_NAME + '-clusters.train'
TEST_FILE_CLUSTERING = MODEL_NAME + '-clusters.test'

CTR_MODEL_FEATURES_CONFIG = {
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
CTR_MODEL_FREE_COEF = -1
CTR_MODEL_FREE_LAMBDA = 100

VW_TREE_FEATURES = ['a', 'b']
LIBFFM_TREE_FEATURES = ['0', '1']


def generate_data(model_name, train_dataset_length, test_dataset_length, formats=None, from_cache=False):
    if not formats:
        formats = ['vw']

    if from_cache:
        real_ctr_model = CTRModel.load(model_name + '.dat')
    else:
        with Timer('init ctr model'):
            real_ctr_model = CTRModel(
                features_config=CTR_MODEL_FEATURES_CONFIG,
                free_coef=CTR_MODEL_FREE_COEF,
                lam=CTR_MODEL_FREE_LAMBDA,
            )
            real_ctr_model.init()

        with Timer('generate datasets'):
            for ouput_format in formats:
                real_ctr_model.generate_log(
                    filename=model_name,
                    format=ouput_format,
                    train_length=train_dataset_length,
                    test_length=test_dataset_length,
                )
                real_ctr_model.generate_log(
                    filename=model_name,
                    format=ouput_format,
                    train_length=train_dataset_length,
                    test_length=test_dataset_length,
                )
            real_ctr_model.save(model_name + '.dat')

    return real_ctr_model


def experiments(debug=False, clustering_levels=2, clustering_processes=1):
    with Timer('converted datasets'):
        f_clustering = FeatureClustering(clustering_processes)
        f_clustering.cluster(TRAIN_FILE + '.vw', VW_TREE_FEATURES, parser=parse_vw_line)
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

        f_clustering = FeatureClustering(clustering_processes)
        f_clustering.cluster(TRAIN_FILE + '.libffm', LIBFFM_TREE_FEATURES, parser=parse_libffm_line)
        f_clustering.convert_log(
            TRAIN_FILE + '.libffm',
            TRAIN_FILE_CLUSTERING + '.libffm',
            {'mode': 'part_tree', 'levels': clustering_levels},
            parser=parse_libffm_line,
            composer=compose_libffm_line,
        )
        f_clustering.convert_log(
            TEST_FILE + '.libffm',
            TEST_FILE_CLUSTERING + '.libffm',
            {'mode': 'part_tree', 'levels': clustering_levels},
            parser=parse_libffm_line,
            composer=compose_libffm_line,
        )

    results = {}

    with Timer('LibFFMClassifier'):
        ctr_model = LibFFMClassifier(debug=debug)
        ctr_model.train(TRAIN_FILE + '.libffm')
        train_ll = ctr_model.test(TRAIN_FILE + '.libffm')
        test_ll = ctr_model.test(TEST_FILE + '.libffm')
        results['LibFFMClassifier'] = {
            'train': train_ll,
            'test': test_ll,
        }

        log.info('LibFFMClassifier LL train - %s', train_ll)
        log.info('LibFFMClassifier LL test - %s', test_ll)

    with Timer('LibFFMClassifierClustering'):
        ctr_model = LibFFMClassifier(debug=debug)
        ctr_model.train(TRAIN_FILE_CLUSTERING + '.libffm')
        train_ll = ctr_model.test(TRAIN_FILE_CLUSTERING + '.libffm')
        test_ll = ctr_model.test(TEST_FILE_CLUSTERING + '.libffm')
        results['LibFFMClassifierClustering'] = {
            'train': train_ll,
            'test': test_ll,
        }

        log.info('LibFFMClassifierClustering LL train - %s', train_ll)
        log.info('LibFFMClassifierClustering LL test - %s', test_ll)

    with Timer('LogisticVWClassifier'):
        ctr_model = LogisticVWClassifier(
            debug=debug,
            passes=100,
            bit_precision=13,
            q=['ab'],
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
            passes=100,
            bit_precision=13,
            q=['ab'],
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
            passes=100,
            bit_precision=13,
            q=['ab'],

            ftrl=True,
            ftrl_alpha=0.1,
            ftrl_beta=0.0,
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
            passes=100,
            bit_precision=13,
            q=['ab'],

            ftrl=True,
            ftrl_alpha=0.1,
            ftrl_beta=0.0,
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
    results_llr = {}
    for i in xrange(NUMBER_OF_EXPERIMENTS):
        ctr_model = generate_data(
            model_name=MODEL_NAME,
            train_dataset_length=TRAIN_DATASET_LENGTH,
            test_dataset_length=TEST_DATASET_LENGTH,
            formats=['vw', 'libffm'],
        )

        ll = ctr_model.loglikelihood()
        ll0 = ctr_model.loglikelihood0()

        log.info('RealCtr LL = %s', ll)
        log.info('RealCtr LL0 = %s', ll0)
        log.info('RealCtr LLr = %s', ll0 - ll)

        results = experiments(
            debug=DEBUG,
            clustering_levels=CLUSTERING_LEVELS,
            clustering_processes=NUMBER_OF_PROCESSES,
        )

        for classifier, classifier_results in results.iteritems():
            results_llr_classifier = results_llr.setdefault(classifier, [])
            results_llr_classifier.append({
                'train': ll0 - classifier_results['train'],
                'test': ll0 - classifier_results['test'],
            })

    for classifier, classifier_results in results_llr.iteritems():
        train_llrs = [classifier_result['train'] for classifier_result in classifier_results]
        test_llrs = [classifier_result['test'] for classifier_result in classifier_results]

        av_train_llr = np.mean(train_llrs)
        av_test_llr = np.mean(test_llrs)
        std_train_llr = np.std(train_llrs)
        std_test_llr = np.std(test_llrs)
        log.info('%s average LLr train - %s', classifier, av_train_llr)
        log.info('%s average LLr test - %s', classifier, av_test_llr)
        log.info('%s std LLr train - %s', classifier, std_train_llr)
        log.info('%s std LLr test - %s', classifier, std_test_llr)


if __name__ == '__main__':
    logging_level = logging.DEBUG if DEBUG else logging.INFO
    load_dict_config(LOGGING, level=logging_level, filename=LOG_FILENAME)
    main()
