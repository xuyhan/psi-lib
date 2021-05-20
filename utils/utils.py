def merge_ops(dict1, dict2):
    dict3 = {k: v for k, v in dict1.items()}
    for k, v in dict2.items():
        if k in dict3:
            dict3[k] += v
        else:
            dict3[k] = v
    return dict3


def init_ops():
    return {
        'addPC': 0,
        'addCC': 0,
        'mulPC': 0,
        'mulCC': 0,
        'rots': 0
    }


def to_str(ops):
    s = ''
    for k, v in ops.items():
        if v != 0:
            s += '({k},{v})'.format(k=k, v=v)
    return s
