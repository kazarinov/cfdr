# -*- coding: utf-8 -*-
import os
import errno
import logging
import time
import math

import scipy as sp

default_logger = logging.getLogger(__name__)


class Timer(object):
    def __init__(self, name='-', logger=None):
        self.name = name
        self.logger = logger or default_logger

    def __enter__(self):
        self.start = time.time()
        self.logger.debug("%s: started", self.name)

    def __exit__(self, exc_type, exc_value, traceback):
        end = time.time()
        self.logger.info("%s: done for %0.3f seconds", self.name, end - self.start)
        return False


def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e:  # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise  # re-raise exception if a different error occured


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


def traverse_tree(rootnode):
    thislevel = [rootnode]
    while thislevel:
        nextlevel = list()
        for n in thislevel:
            print n.value,
            if n.left: nextlevel.append(n.left)
            if n.right: nextlevel.append(n.right)
        print
        thislevel = nextlevel
