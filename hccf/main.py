#!/usr/bin/env python
# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import logging
import os.path

from hccf.clustering import FeatureClustering
from hccf.utils.tools import get_parser, get_composer
from hccf.settings import LOGGING
from hccf.utils.logs import load_dict_config


FORMATS = ['vw', 'libffm']  # возможные форматы входных данных
DEFAULT_FORMAT = 'vw'


def cluster(input, model, format, features, slices, processes=1):
    fc = FeatureClustering(processes)
    parser = get_parser(format)
    fc.cluster(
        input_logfile=input,
        tree_features=features,
        slice_features=slices,
        parser=parser,
    )
    fc.save(model)


def convert(input, output, model, type, levels, parser, composer):
    fc = FeatureClustering.load(model)
    parser = get_parser(parser)
    composer = get_composer(composer)
    output_params = {
        'type': type,
        'levels': levels,
    }
    fc.convert_log(
        input_logfile=input,
        output_logfile=output,
        output_params=output_params,
        parser=parser,
        composer=composer,
    )


def graph(model, feature):
    fc = FeatureClustering.load(model)
    print fc.gml_graph(feature)


def main():
    argparser = ArgumentParser(description='Agglomerative Hierarchical Clustering')
    subparsers = argparser.add_subparsers(help='sub-command help', dest='command')

    # clustering command
    parser_cluster = subparsers.add_parser('cluster', help='clustering')
    parser_cluster.add_argument('--debug', help="debug mode", action='store_true')
    parser_cluster.add_argument('-p', '--processes', dest="processes", help="number of processes", type=int, default=1)
    parser_cluster.add_argument(
        '--format',
        dest='format',
        help="format of input and output file",
        default=DEFAULT_FORMAT,
        choices=FORMATS,
    )
    parser_cluster.add_argument('-i', '--input', dest="input", help="input file", required=True)
    parser_cluster.add_argument('-m', '--model', dest="model", help="model file", required=True)
    parser_cluster.add_argument(
        '-f', '--features',
        dest='features',
        help='features to clustering',
        nargs='*',
        required=True,
    )
    parser_cluster.add_argument(
        '-s', '--slices',
        dest='slices',
        help='slices for clustering',
        nargs='*',
    )

    # convert command
    parser_convert = subparsers.add_parser('convert', help='convert')
    parser_convert.add_argument('--debug', help="debug mode", action='store_true')
    parser_convert.add_argument('-i', '--input', dest="input", help="input file", required=True)
    parser_convert.add_argument('-o', '--output', dest="output", help="output file", required=True)
    parser_convert.add_argument(
        '-t', '--type',
        dest="type",
        help="convert type",
        default='full_tree',
        choices=['full_tree', 'tree_without_leaves', 'part_tree'],
    )
    parser_convert.add_argument('-l', '--levels', dest="levels", help="levels", default=None)
    parser_convert.add_argument('-m', '--model', dest="model", help="model file", required=True)
    parser_convert.add_argument(
        '--parser',
        dest='parser',
        help="parser of input file",
        default=DEFAULT_FORMAT,
        choices=FORMATS,
    )
    parser_convert.add_argument(
        '--composer',
        dest='composer',
        help="composer of output file",
        default=DEFAULT_FORMAT,
        choices=FORMATS,
    )

    # graph command
    parser_graph = subparsers.add_parser('graph', help='GML graph')
    parser_graph.add_argument('-m', '--model', dest="model", help="model file", required=True)
    parser_graph.add_argument('-f', '--feature', dest="feature", help="feature name", required=True)


    options = argparser.parse_args()
    logging_level = logging.DEBUG if getattr(options, 'debug', False) else logging.INFO
    load_dict_config(LOGGING, logging_level)

    if options.command == 'cluster':
        if not os.path.isfile(options.input):
            argparser.error('no input file `%s` found' % options.input)

        cluster(
            input=options.input,
            model=options.model,
            format=options.format,
            features=options.features,
            slices=options.slices,
            processes=options.processes,
        )
    elif options.command == 'convert':
        if not os.path.isfile(options.model):
            argparser.error('no model file `%s` found' % options.model)

        convert(
            input=options.input,
            output=options.output,
            model=options.model,
            type=options.type,
            levels=options.levels,
            parser=options.parser,
            composer=options.composer,
        )
    elif options.command == 'graph':
        if not os.path.isfile(options.model):
            argparser.error('no model file `%s` found' % options.model)

        graph(
            model=options.model,
            feature=options.feature,
        )
    else:
        argparser.error('no command `%s` found' % options.command)


if __name__ == '__main__':
    main()
