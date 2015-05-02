# -*- coding: utf-8 -*-
import math
import scipy as sp


def sigmoid(z):
    s = 1.0 / (1.0 + math.exp(-z))
    return s


def log_loss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    ll = sum(act * sp.log(pred.astype(float)) + sp.subtract(1, act.astype(float)) * sp.log(
        sp.subtract(1, pred.astype(float))))
    ll = ll * -1.0 / len(act)
    return ll


def loglikelihood(shows, clicks):
    if clicks == 0 or shows == 0 or clicks == shows:
        return 0

    ctr = float(clicks) / shows
    return -1 * (clicks * math.log(ctr) + (shows - clicks) * math.log(1 - ctr))
