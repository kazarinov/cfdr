# -*- coding: utf-8 -*-


def make_vw_command(predictions='/dev/stdout',
                    quiet=True,
                    save_resume=True,
                    q_colon=None,
                    **kwargs):
    """Construct a command line for VW, with each named argument corresponding
    to a VW option.
    Single character keys are mapped to single-dash options,
    e.g. 'b=20' yields '-b 20',
    while multiple character keys map to double-dash options:
        'quiet=True' yields '--quiet'
    Boolean values are interpreted as flags: present if True, absent if False.
    All other values are treated as option arguments, as in the -b example above.
    If an option argument is a list, that option is repeated multiple times,
    e.g. 'q=['ab', 'bc']' yields '-q ab -q bc'
    q_colon is handled specially, mapping to '--q:'.
    Run 'vw -h' for a listing of most options.
    Defaults are well-suited for use with Wabbit Wappa:
    vw --predictions /dev/stdout --quiet --save_resume
    NOTE: This function makes no attempt to validate the inputs or
    ensure they are compatible with Wabbit Wappa.
    Outputs a command line string.
    """
    if q_colon:
        kwargs['q:'] = q_colon
    kwargs['predictions'] = predictions
    kwargs['quiet'] = quiet
    kwargs['save_resume'] = save_resume
    return make_command('vw', **kwargs)


def make_command(program, *args, **kwargs):
    command_parts = [program]
    for key, value in kwargs.items():
        if len(key) == 1:
            option = '-{}'.format(key)
        else:
            option = '--{}'.format(key)
        if value is False:
            continue
        elif value is True:
            arg_list = [option]
        elif isinstance(value, basestring):
            arg_list = ['{} {}'.format(option, value)]
        elif hasattr(value, '__getitem__'):  # Listlike value
            arg_list = ['{} {}'.format(option, subvalue) for subvalue in value]
        else:
            arg_list = ['{} {}'.format(option, value)]
        command_parts.extend(arg_list)
    command_parts.extend(args)
    command_parts = ' '.join(command_parts)
    return command_parts


def parse_vw_line(line):
    params = {}

    parts = map(lambda s: s.strip(), line.strip().split('|'))
    label = parts.pop(0)
    params['label'] = int(label)
    params['features'] = []
    for part in parts:
        name, value = part.split(' ')
        try:
            value = int(value)
        except ValueError:
            pass

        params['features'].append((name, value))

    return params


def compose_vw_line(label, features):
    label = 1 if label > 0 else -1
    vw_feature_values = map(lambda t: '%s %s ' % (t[0], ' '.join(map(str, t[1])) if isinstance(t[1], list) else t[1]), features)
    return '%s |%s' % (label, '|'.join(vw_feature_values))


def parse_libffm_line(line):
    # не парсит разные индексы одного филда
    params = {}

    parts = map(lambda s: s.strip(), line.strip().split(' '))
    label = parts.pop(0)
    params['label'] = int(label)
    params['features'] = []
    for part in parts:
        field, index, value = part.split(':')
        try:
            field = str(field)
            index = int(index)
            value = int(value)
        except ValueError:
            pass

        params['features'].append((field, index))

    return params


def compose_libffm_line(label, features):
    # features: [(name, feature_value), ...], feature_value = int | [int, int, ...]
    label = 1 if label > 0 else 0
    feature_values = []
    for feature_index, (feature_name, feature_value) in enumerate(features):
        if isinstance(feature_value, list):
            feature_values.extend(map(lambda x: '%s:%s:1' % (feature_index, x), feature_value))
        else:
            feature_values.append('%s:%s:1' % (feature_index, feature_value))

    return '%s %s' % (label, ' '.join(feature_values))
