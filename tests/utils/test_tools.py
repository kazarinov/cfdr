# -*- coding: utf-8 -*-
import pytest
from hccf.utils import tools


def test_make_vw_command_no_extra_args():
    command = tools.make_vw_command(
        predictions='predictions.txt',
        quiet=True,
        save_resume=True,
    )
    expected_command = './vowpal_wabbit/vowpalwabbit/vw --save_resume --predictions predictions.txt --quiet'
    assert command == expected_command


def test_make_vw_command_extra_args():
    command = tools.make_vw_command(
        predictions='predictions.txt',
        quiet=True,
        save_resume=True,
        a='test1',
        b=2,
        c=True,
        d=[0, 1],
    )
    expected_command = './vowpal_wabbit/vowpalwabbit/vw -a test1 --save_resume -c -b 2 -d 0 -d 1 --quiet --predictions predictions.txt'
    assert command == expected_command


def test_make_command_no_args():
    command = tools.make_command('program')
    expected_command = 'program'
    assert command == expected_command


def test_make_command_with_args():
    command = tools.make_command(
        'program',
        'arg1',
        'arg2',
        a='test1',
        b=2,
        c=True,
        d=[0, 1],
    )
    expected_command = 'program -a test1 -c -b 2 -d 0 -d 1 arg1 arg2'
    assert command == expected_command


def test_parse_vw_line_one_value_ok():
    results = tools.parse_vw_line('1 |a 1 |b 2')
    assert results['label'] == 1
    assert results['features'] == [('a', 1), ('b', 2)]


def test_parse_vw_line_multi_values_ok():
    results = tools.parse_vw_line('-1 |a 1 2 |b 2 3')
    assert results['label'] == -1
    assert results['features'] == [('a', ['1', '2']), ('b', ['2', '3'])]


def test_parse_vw_line_empty_line():
    with pytest.raises(ValueError):
        tools.parse_vw_line('')


def test_parse_vw_line_invalid_format():
    with pytest.raises(ValueError):
        tools.parse_vw_line('1:1:1 1:2:1 2:3:1')


def test_compose_vw_line_0():
    line = tools.compose_vw_line(0, [('a', 1), ('b', 2)])
    assert line == '-1 |a 1 |b 2'


def test_compose_vw_line_1():
    line = tools.compose_vw_line(1, [('a', 1), ('b', 2)])
    assert line == '1 |a 1 |b 2'


def test_parse_libffm_line_one_value_ok():
    results = tools.parse_libffm_line('1 1:1:1 2:3:1')
    assert results['label'] == 1
    assert results['features'] == [('1', 1), ('2', 3)]


def test_parse_libffm_line_multi_values_ok():
    results = tools.parse_libffm_line('0 1:1:1 1:2:1 2:3:1')
    assert results['label'] == 0
    assert results['features'] == [('1', 1), ('1', 2), ('2', 3)]


def test_parse_libffm_line_empty_line():
    with pytest.raises(ValueError):
        tools.parse_libffm_line('')


def test_parse_libffm_line_invalid_format():
    with pytest.raises(ValueError):
        tools.parse_libffm_line('1 |a 1 |b 2')


def test_compose_libffm_line_0():
    line = tools.compose_libffm_line(0, [('a', 1), ('b', 2)])
    assert line == '0 0:1:1 1:2:1'


def test_compose_libffm_line_1():
    line = tools.compose_libffm_line(1, [('a', 1), ('b', 2)])
    assert line == '1 0:1:1 1:2:1'


def test_get_parser_vw():
    tools.get_parser('vw') == tools.parse_vw_line


def test_get_parser_libffm():
    tools.get_parser('libffm') == tools.parse_libffm_line


def test_get_parser_invalid():
    with pytest.raises(ValueError):
        tools.get_parser('invalid')


def test_get_composer_vw():
    tools.get_composer('vw') == tools.compose_vw_line


def test_get_composer_libffm():
    tools.get_composer('libffm') == tools.compose_libffm_line


def test_get_composer_invalid():
    with pytest.raises(ValueError):
        tools.get_composer('invalid')
