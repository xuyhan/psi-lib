def merge_ops(dict1, dict2):
    dict3 = {k: v for k, v in dict1.items()}
    for k, v in dict2.items():
        if k in dict3:
            dict3[k] += v
        else:
            dict3[k] = v
    return dict3