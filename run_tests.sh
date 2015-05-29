#!/bin/bash

PYTHONPATH=.:$PYTHONPATH py.test tests $@
