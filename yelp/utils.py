import math
import random
import typing

import numpy as np
import torch


def to_gpu(gpu, var):
    if gpu:
        return var.cuda()
    return var


def batchify(data, bsz, shuffle=False, gpu=False):
    if shuffle:
        random.shuffle(data)

    nbatch = int(math.ceil(len(data) / bsz))
    batches = []

    for i in range(nbatch):
        # Pad batches to maximum sequence length in batch
        batch = data[i * bsz: (i + 1) * bsz]

        # subtract 1 from lengths b/c includes BOTH starts & end symbols
        words = batch
        lengths = [len(x) - 1 for x in words]

        # sort items by length (decreasing)
        batch, lengths = length_sort(batch, lengths)
        words = batch

        # source has no end symbol
        source = [x[:-1] for x in words]
        # target has no start symbol
        target = [x[1:] for x in words]

        # find length to pad to
        maxlen = max(lengths)
        for x, y in zip(source, target):
            zeros = (maxlen - len(x)) * [0]
            x += zeros
            y += zeros

        source = torch.LongTensor(np.array(source))
        target = torch.LongTensor(np.array(target)).view(-1)

        batches.append((source, target, lengths))
    return batches


def length_sort(items, lengths, descending=True):
    """In order to use pytorch variable length sequence package"""
    items = list(zip(items, lengths))
    items.sort(key=lambda x: x[1], reverse=True)
    items, lengths = zip(*items)
    return list(items), list(lengths)


def format_epoch(epoch):
    if epoch is None:
        return 'final'
    else:
        return '{:03d}'.format(epoch)


class MultipleInheritanceNamedTupleMeta(typing.NamedTupleMeta):
    # Makes extending NamedTuple possible
    # https://stackoverflow.com/questions/50367661/customizing-typing-namedtuple
    def __new__(mcls, typename, bases, ns):
        if typing.NamedTuple in bases:
            base = super().__new__(mcls, '_base_' + typename, bases, ns)
            bases = (base, *(b for b in bases if not isinstance(b, typing.NamedTuple)))
        return super(typing.NamedTupleMeta, mcls).__new__(mcls, typename, bases, ns)


class NamedTupleFromArgs(metaclass=MultipleInheritanceNamedTupleMeta):
    @classmethod
    def from_args(cls, args, **additional_fields):
        values = []
        for field_name in cls.__annotations__:
            try:
                value = getattr(args, field_name)
            except AttributeError:
                value = additional_fields[field_name]
            values.append(value)
        return cls(*values)
