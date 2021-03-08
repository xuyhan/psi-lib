import numpy as np

def load_data(name):
    weights = np.load('data/' + name + '_weights.npy', allow_pickle=True)
    test = np.load('data/' + name + '_testX.npy')
    preds_base = np.load('data/' + name + '_preds.npy')
    outputs_base = np.load('data/' + name + '_outs.npy')

    return weights, test, preds_base, outputs_base