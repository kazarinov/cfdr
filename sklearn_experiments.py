# -*- coding: utf-8 -*-
import logging

import pandas as pd
import numpy as np

from cfdr.experiments.ctr_model import CTRModel
from cfdr.utils.helpers import Timer
from sklearn.feature_extraction import FeatureHasher
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC


log = logging.getLogger(__name__)


FEATURES_CONFIG = {
    'a': {
        'count': 64,
        'loc': 0.0,
        'scale': 0.5,
        'type': 'tree',
    },
    'b': {
        'count': 50,
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


def clean_data(filename):
    preprocessor = Pipeline([
        ('fh', FeatureHasher(n_features=2 ** 13, input_type='string', non_negative=False)),
    ])

    train_data = pd.read_table(filename, sep=',', chunksize=10000)
    train_data = train_data.read()
    y_train = train_data['click']
    train_data.drop(['click'], axis=1, inplace=True)  # remove id and click columns
    x_train = np.asarray(train_data.astype(str))
    y_train = np.asarray(y_train).ravel()
    x_train = preprocessor.fit_transform(x_train).toarray()

    return x_train, y_train


def clean_data_chunked(filename):
    preprocessor = Pipeline([
        ('fh', FeatureHasher(n_features=2 ** 13, input_type='string', non_negative=False)),
    ])

    train_data = pd.read_table(filename, sep=',', chunksize=1000)
    for train_data_chunk in train_data:
        print 'process chunk'
        y_train = train_data_chunk['click']
        train_data_chunk.drop(['click'], axis=1, inplace=True)  # remove id and click columns
        x_train = np.asarray(train_data_chunk.astype(str))
        y_train = np.asarray(y_train).ravel()
        x_train = preprocessor.fit_transform(x_train).toarray()
        yield x_train, y_train


def create_dataset(model='sklearn-clicklog', from_cache=False, train_dataset_length=100000, test_dataset_length=100000):
    train_filename = model + '.train.csv'
    test_filename = model + '.test.csv'

    if from_cache:
        real_ctr_model = CTRModel.load(model + '.dat')
    else:
        with Timer('init real model'):
            real_ctr_model = CTRModel(FEATURES_CONFIG, free_coef=-1, lam=100)
            real_ctr_model.init()

        with Timer('generate clicklog'):
            real_ctr_model.generate_log(
                filename=model,
                format='csv',
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

    return train_filename, test_filename


def ctr_gbdt(model='sklearn-clicklog', from_cache=False, train_dataset_length=100000, test_dataset_length=100000):
    TRAIN_FILE, TEST_FILE = create_dataset(model, from_cache, train_dataset_length, test_dataset_length)

    prediction_model = GradientBoostingClassifier(
        loss='deviance',
        learning_rate=0.1,
        n_estimators=30,
        subsample=1.0,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=5,
    )

    x_train, y_train = clean_data(TRAIN_FILE)
    x_test, y_test = clean_data(TEST_FILE)

    with Timer('fit model'):
        prediction_model.fit(x_train, y_train)

    with Timer('evaluate model'):
        y_prediction_train = prediction_model.predict_proba(x_train)
        y_prediction_test = prediction_model.predict_proba(x_test)

        loss_train = log_loss(y_train, y_prediction_train)
        loss_test = log_loss(y_test, y_prediction_test)

    print 'loss_train: %s' % loss_train
    print 'loss_test: %s' % loss_test


def ctr_pca_sgd(model='sklearn-clicklog', from_cache=False, train_dataset_length=100000, test_dataset_length=100000):
    TRAIN_FILE, TEST_FILE = create_dataset(model, from_cache, train_dataset_length, test_dataset_length)

    prediction_model = SGDClassifier(
        loss='log',
        n_iter=200,
        alpha=.0000001,
        penalty='l2',
        learning_rate='invscaling',
        power_t=0.5,
        eta0=4.0,
        shuffle=True,
        n_jobs=-1,
    )

    x_train, y_train = clean_data(TRAIN_FILE)
    x_test, y_test = clean_data(TEST_FILE)

    pca = PCA(n_components=100)
    pca.fit(x_train)

    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)

    with Timer('fit model'):
        prediction_model.fit(x_train, y_train)

    with Timer('evaluate model'):
        y_prediction_train = prediction_model.predict_proba(x_train)
        y_prediction_test = prediction_model.predict_proba(x_test)

        loss_train = log_loss(y_train, y_prediction_train)
        loss_test = log_loss(y_test, y_prediction_test)

    print 'loss_train: %s' % loss_train
    print 'loss_test: %s' % loss_test


def ctr_svm(model='sklearn-clicklog', from_cache=False, train_dataset_length=100000, test_dataset_length=100000):
    """
    Doesn't work
    """
    TRAIN_FILE, TEST_FILE = create_dataset(model, from_cache, train_dataset_length, test_dataset_length)

    prediction_model = LinearSVC(
        penalty='l1',
        loss='squared_hinge',
        dual=False,
        tol=0.0001,
        C=1.0,
        multi_class='ovr',
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        verbose=1,
        random_state=None,
        max_iter=1000,
    )


    x_train, y_train = clean_data(TRAIN_FILE)
    x_test, y_test = clean_data(TEST_FILE)

    with Timer('fit model'):
        prediction_model.fit(x_train, y_train)

    with Timer('evaluate model'):
        y_prediction_train = prediction_model.predict_proba(x_train)
        y_prediction_test = prediction_model.predict_proba(x_test)

        loss_train = log_loss(y_train, y_prediction_train)
        loss_test = log_loss(y_test, y_prediction_test)

    print 'loss_train: %s' % loss_train
    print 'loss_test: %s' % loss_test


if __name__ == '__main__':
    # ctr_gbdt(
    #     from_cache=False,
    #     train_dataset_length=100000,
    #     test_dataset_length=100000,
    # )

    # ctr_pca_sgd(
    #     from_cache=False,
    #     train_dataset_length=100000,
    #     test_dataset_length=100000,
    # )

    # ctr_svm(
    #     model='sklearn-clicklog',
    #     from_cache=False,
    #     train_dataset_length=100000,
    #     test_dataset_length=100000,
    # )

    ctr_ftrl(
        model='sklearn-clicklog',
        from_cache=False,
        train_dataset_length=100000,
        test_dataset_length=100000,
    )
