# -*- coding: utf-8 -*-
from functools import wraps

import os
import sys
import signal
import logging
import itertools
import datetime
import traceback


log = logging.getLogger(__name__)


class FilterOut(object):
    def __init__(self, name):
        self.name = name

    def filter(self, record):
        return not record.name.startswith(self.name)


def load_dict_config(settings, level=None, filename=None, filename_error=None):
    formatters = dict()
    for name, formatter in settings["formatters"].iteritems():
        if 'datefmt' in formatter:
            formatters[name] = logging.Formatter(fmt=formatter['format'], datefmt=formatter['datefmt'])
        else:
            formatters[name] = logging.Formatter(fmt=formatter['format'])

    handlers = dict()
    for name, hndlr in settings["handlers"].iteritems():
        if hndlr['class'] == 'logging.FileHandler':
            if hndlr['level'] == logging.ERROR and filename_error:
                handler = logging.FileHandler(hndlr['filename'] if hndlr['filename'] else filename_error)
            else:
                handler = logging.FileHandler(hndlr['filename'] if hndlr['filename'] else filename)

            handler.setLevel(level if level else hndlr['level'])
            if hndlr['formatter'] in formatters:
                handler.setFormatter(formatters[hndlr['formatter']])
            handlers[name] = handler
            for filter_out in hndlr.get('filters', []):
                handler.addFilter(FilterOut(filter_out))

        elif hndlr['class'] == 'logging.StreamHandler':
            handler = logging.StreamHandler(hndlr.get("stream", sys.stderr))
            handler.setLevel(level if level else hndlr['level'])
            if hndlr['formatter'] in formatters:
                handler.setFormatter(formatters[hndlr['formatter']])
            handlers[name] = handler

    if "root" in settings:
        lgr = settings["root"]
        if 'level' in lgr:
            logging.root.setLevel(level if level else lgr['level'])
        if 'handlers' in lgr:
            for handler_name in lgr['handlers']:
                if handler_name in handlers:
                    logging.root.addHandler(handlers[handler_name])

    if "loggers" in settings:
        for name, lgr in settings["loggers"].iteritems():
            logger = logging.getLogger(name)
            if 'level' in lgr:
                logger.setLevel(lgr['level'])
            if 'propagate' in lgr:
                logger.propagate = lgr['propagate']
            if 'handlers' in lgr:
                for handler_name in lgr['handlers']:
                    if handler_name in handlers:
                        logger.addHandler(handlers[handler_name])

    set_logs_reloader_on_usr1()


def logs_reloader(*args, **kwargs):
    logging.info("reloading logs after SIGUSR1")
    reopen_all_log_files()
    logging.info("reloaded logs after SIGUSR1")


def set_logs_reloader_on_usr1():
    signal.signal(signal.SIGUSR1, logs_reloader)


def loggers():
    """ get list of all loggers """
    root = logging.root
    existing = root.manager.loggerDict.keys()
    return [logging.getLogger(name) for name in existing]


def reopen_root_logger():
    for handler in logging.root.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.acquire()
            try:
                if handler.stream:
                    handler.stream.close()
                    handler.stream = open(handler.baseFilename,
                                          handler.mode)
            finally:
                handler.release()


def reopen_all_log_files():
    rootfile = None
    for handler in logging.root.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.acquire()
            try:
                if handler.stream:
                    handler.stream.close()
                    handler.stream = open(handler.baseFilename,
                                          handler.mode)
                    if not rootfile:
                        rootfile = handler.stream
            finally:
                handler.release()

    if rootfile:
        # if an error log file is set redirect stdout & stderr to
        # this log file.
        for stream in sys.stdout, sys.stderr:
            stream.flush()

        os.dup2(rootfile.fileno(), sys.stdout.fileno())
        os.dup2(rootfile.fileno(), sys.stderr.fileno())

    for log in loggers():
        for handler in log.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.acquire()
                try:
                    if handler.stream and handler.stream != rootfile:
                        handler.stream.close()
                        handler.stream = open(handler.baseFilename,
                                              handler.mode)
                finally:
                    handler.release()


def show_loggers(message=None):
    if message:
        print message
    for handler in logging._handlerList:
        print "handler", str(handler), "level:", handler.level, "check:"
        if isinstance(handler, logging.StreamHandler):
            handler.stream.write("is working\n")
            handler.stream.flush()
    for handler in logging._handlerList:
        print "handler", str(handler), "level:", handler.level, "check:"
        if isinstance(handler, logging.StreamHandler):
            handler.stream.write("is working\n")
            handler.stream.flush()
    print "root handlers:", logging.root.handlers, "root level:", logging.root.level
    for loggername in logging.root.manager.loggerDict.iterkeys():
        lgr = logging.getLogger(loggername)
        print "logger's %s (%s) handlers:" % (loggername, str(lgr)), lgr.handlers, \
            "propagate: %s" % lgr.propagate, "level: %s" % lgr.level
    print


def check_file_handler(level, filename, mode='a'):
    fullpath = os.path.abspath(filename)
    for handler in logging._handlerList:
        if isinstance(handler, logging.FileHandler):
            if handler.baseFilename == fullpath and handler.level == level and handler.mode == mode:
                return handler


def log_exception_to(log_method):
    def apply_log_exception_to(func):
        '''декоратор, протоколирующий все возникающие исключения в методе'''

        @wraps(func)
        def wrapper(*args, **kw):
            try:
                return func(*args, **kw)
            except Exception, e:
                log_method(traceback.format_exc())
                raise

        return wrapper

    return apply_log_exception_to


def do_call_log_time_warn(func, args, kw, log_method=log.debug, name_prefix=None, timeout_warn=None):
    if name_prefix:
        name = '%s.%s' % (name_prefix, func.__name__)
    else:
        name = func.__name__

    args_s = ', '.join(itertools.chain((repr(i) for i in args), ('%s=%s' % (k, repr(v)) for (k, v) in kw.iteritems())))

    start_time = datetime.datetime.now()
    try:
        return func(*args, **kw)
    finally:
        func_time = datetime.datetime.now() - start_time
        log_method('%s ended, execution time=%s' %
                   (name, func_time))
        if timeout_warn is not None and func_time > datetime.timedelta(0, 0, timeout_warn * 1000000):
            log.warn('%s(%s) ended, execution time over %ss = %s',
                     name, args_s, timeout_warn, func_time)


def call_log_to(log_method=log.debug, name_prefix=None, timeout_warn=None):
    '''
    log_method -- метод, который вызывается для протоколирования
    timeout_warn -- время в секундах, превышение которого генерирует варнинг,
                    None если варнинг не нужен
    '''

    def apply_call_log(func):
        @wraps(func)
        def wrapper(*args, **kw):
            return do_call_log_time_warn(func, args, kw, log_method, name_prefix, timeout_warn)

        return wrapper

    return apply_call_log


call_log = call_log_to()

