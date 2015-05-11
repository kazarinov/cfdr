# -*- coding: utf-8 -*-
import os
import errno
import logging
import time

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


class GetList(list):
    """Allows lists to be interfaced with get as dicts would be.
    Unfortunatey, this workaround is slower.
    'default' is not used since lists are initalized with defaults
    already.
    """
    def get(self, key, default):
        return self[key]
