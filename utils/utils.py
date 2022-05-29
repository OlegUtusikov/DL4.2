import logging
import random
import sys

import numpy as np
import torch


def pad(seq, max_len):
    if len(seq) < max_len:
        seq = seq + ['<pad>'] * (max_len - len(seq))

    return seq[0:max_len]


def to_indexes(vocab, words):
    return [vocab.stoi[w] for w in words]


def get_logger():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)
    return root


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)