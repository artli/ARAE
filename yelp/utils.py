import json
import os
import logging
import math
import random
import string
from collections import Counter

import numpy as np
import torch
from nltk import word_tokenize, sent_tokenize


PAD_WORD = "<pad>"
EOS_WORD = "<eos>"
BOS_WORD = "<bos>"
UNK = "<unk>"


def load_kenlm():
    global kenlm
    import kenlm


def to_gpu(gpu, var):
    if gpu:
        return var.cuda()
    return var


class Dictionary:
    _FIRST_WORDS = [PAD_WORD, BOS_WORD, EOS_WORD, UNK]

    def __init__(self, word2idx):
        self.word2idx = word2idx
        self.idx2word = {v: k for k, v in word2idx.items()}

    @classmethod
    def from_files(cls, filenames, lowercase=True, max_size=None):
        counter = Counter()
        for path in filenames:
            with open(path) as f:
                for line in f:
                    if lowercase:
                        line = line.lower()
                    for word in line.strip().split(' '):
                        counter[word] += 1

        words, _ = zip(*counter.most_common(max_size))
        word2idx = dict(cls._FIRST_WORDS + words)
        return cls(word2idx)

    @classmethod
    def load(cls, working_dir):
        with open('{}/vocab.json'.format(working_dir)) as f:
            word2idx = json.load(f)
        return cls(word2idx)

    def save(self, working_dir):
        with open('{}/vocab.json'.format(working_dir), 'w') as f:
            json.dump(self.word2idx, f)

    def __len__(self):
        return len(self.word2idx)


class Preprocessor:
    def __init__(self, dictionary: Dictionary):
        self.dictionary = dictionary

    def words_to_ids(self, sentence, maxlen=0):
        if maxlen > 0:
            sentence = sentence[:maxlen]
        words = [BOS_WORD] + sentence + [EOS_WORD]
        vocab = self.dictionary.word2idx
        unk_id = vocab[UNK]
        return [vocab.get(word, unk_id) for word in words]

    def text_to_ids(self, text, maxlen=0):
        text = text.lower()
        sentences = [word_tokenize(sentence) for sentence in sent_tokenize(text)]
        return [self.words_to_ids(sentence, maxlen=maxlen) for sentence in sentences]

    def text_to_batch(self, text, maxlen=0):
        encoded = self.text_to_ids(text, maxlen=maxlen)
        return batchify(encoded, len(encoded))[0]

    def batch_to_sentences(self, batch):
        sentences = [[self.dictionary.idx2word[id] for id in sentence]
                     for sentence in batch]
        for sentence in sentences:
            if EOS_WORD in sentence:
                sentence[sentence.index(EOS_WORD):] = []
        return sentences

    def sentence_to_text(self, sentence):
        to_join = []
        for token in sentence:
            if not to_join:
                token = token.capitalize()
            if token not in string.punctuation and '\'' not in token:
                to_join.append(' ')
            to_join.append(token)
        return ''.join(to_join).strip()

    def batch_to_text(self, batch):
        return ' '.join(map(self.sentence_to_text, self.batch_to_sentences(batch)))


class Corpus:
    def __init__(self, source_paths: dict, maxlen, preprocessor: Preprocessor, lowercase=False):
        self.maxlen = maxlen
        self.lowercase = lowercase
        self.data = {
            name: self._tokenize(path, preprocessor)
            for name, path in source_paths.items()}

    def _tokenize(self, path, preprocessor: Preprocessor):
        dropped = 0
        with open(path) as f:
            linecount = 0
            lines = []
            for line in f:
                linecount += 1
                if self.lowercase:
                    line = line.lower()
                words = line.strip().split(' ')
                if len(words) > self.maxlen > 0:
                    dropped += 1
                    continue
                lines.append(preprocessor.words_to_ids(words))

        logging.info('Dropped {} sentences out of {} from {}'.format(dropped, linecount, path))
        return lines


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


def truncate(words):
    # truncate sentences to first occurrence of <eos>
    truncated_sent = []
    for w in words:
        if w != EOS_WORD:
            truncated_sent.append(w)
        else:
            break
    sent = " ".join(truncated_sent)
    return sent


def train_ngram_lm(kenlm_path, data_path, output_path, N):
    """
    Trains a modified Kneser-Ney n-gram KenLM from a text file.
    Creates a .arpa file to store n-grams.
    """
    # create .arpa file of n-grams
    curdir = os.path.abspath(os.path.curdir)

    command = "bin/lmplz -o " + str(N) + " <" + os.path.join(curdir, data_path) + \
              " >" + os.path.join(curdir, output_path)
    os.system("cd " + os.path.join(kenlm_path, 'build') + " && " + command)

    load_kenlm()
    # create language model
    model = kenlm.Model(output_path)

    return model


def get_ppl(lm, sentences):
    """
    Assume sentences is a list of strings (space delimited sentences)
    """
    total_nll = 0
    total_wc = 0
    for sent in sentences:
        words = sent.strip().split()
        score = lm.score(sent, bos=True, eos=False)
        word_count = len(words)
        total_wc += word_count
        total_nll += score
    ppl = 10 ** -(total_nll / total_wc)
    return ppl
