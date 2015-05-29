# -*- coding: utf-8 -*-
import numpy as np
from hccf.utils import mathematics


def test_sigmoid_0():
    assert mathematics.sigmoid(0) == 0.5


def test_log_loss():
    assert mathematics.log_loss(np.array([1, 0]), np.array([0.5, 0.5])) > 0.69


def test_loglikelihood():
    assert mathematics.loglikelihood(0, 0) == 0


def test_loglikelihood_clicks_0():
    assert mathematics.loglikelihood(5, 0) == 0


def test_loglikelihood_shows_0():
    assert mathematics.loglikelihood(0, 5) == 0


def test_loglikelihood_clicks_shows_equal():
    assert mathematics.loglikelihood(5, 5) == 0


def test_loglikelihood_ok():
    assert mathematics.loglikelihood(2, 1) > 0
