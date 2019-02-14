import json
import logging
import os
import string
from collections import Counter
from typing import NamedTuple, Collection, List

from nltk import sent_tokenize, word_tokenize

from utils import NamedTupleFromArgs, TrainingBatch, ModelBatch, batchify, permute


PAD_WORD = "<pad>"
EOS_WORD = "<eos>"
BOS_WORD = "<bos>"
UNK = "<unk>"


class Dictionary:
    _FIRST_WORDS = PAD_WORD, BOS_WORD, EOS_WORD, UNK

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
        word2idx = dict(zip(cls._FIRST_WORDS + words, range(len(cls._FIRST_WORDS) + len(words))))
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

    def sentence_to_ids(self, sentence: List[str], maxlen: int = 0) -> List[int]:
        if maxlen > 0:
            sentence = sentence[:maxlen]
        words = [BOS_WORD] + sentence + [EOS_WORD]
        vocab = self.dictionary.word2idx
        unk_id = vocab[UNK]
        return [vocab.get(word, unk_id) for word in words]

    def text_to_sentences(self, text: str) -> List[List[str]]:
        return [
            word_tokenize(sentence)
            for sentence in sent_tokenize(text.lower())]

    def text_to_model_batch(self, text: str, maxlen: int = 0) -> ModelBatch:
        return [
            self.sentence_to_ids(sentence, maxlen=maxlen)
            for sentence in self.text_to_sentences(text)]

    def text_to_training_batch(self, text: str, maxlen: int = 0) -> TrainingBatch:
        encoded = self.text_to_model_batch(text, maxlen=maxlen)
        return batchify(encoded, len(encoded))[0]

    def model_batch_to_sentences(self, batch: ModelBatch, inverse_permutation=None) -> List[List[str]]:
        sentences = [[self.dictionary.idx2word[id] for id in sentence]
                     for sentence in permute(batch, inverse_permutation)]
        for sentence in sentences:
            if EOS_WORD in sentence:
                sentence[sentence.index(EOS_WORD):] = []
        return sentences

    def sentence_to_text(self, sentence: Collection[str]) -> str:
        to_join = []
        for token in sentence:
            if not to_join:
                token = token.capitalize()
            if token not in string.punctuation and '\'' not in token:
                to_join.append(' ')
            to_join.append(token)
        return ''.join(to_join).strip()

    def model_batch_to_text(self, batch: ModelBatch, inverse_permutation=None) -> str:
        return ' '.join(map(self.sentence_to_text,
                            self.model_batch_to_sentences(batch, inverse_permutation=inverse_permutation)))


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
                lines.append(preprocessor.sentence_to_ids(words))

        logging.info('Dropped {} sentences out of {} from {}'.format(dropped, linecount, path))
        return lines


class DataConfig(NamedTuple, NamedTupleFromArgs):
    working_dir: str
    data_dir: str
    vocab_path: str
    vocab_size: int
    maxlen: int
    lowercase: bool
    batch_size: int


class Data:
    def __init__(self, config: DataConfig, dictionary: Dictionary = None):
        self.config = config

        def to_path_dict(names):
            return {
                name: os.path.join(config.data_dir, name + '.txt')
                for name in names}
        train_names = 'train1', 'train2'
        valid_names = 'valid1', 'valid2'

        if dictionary is None:
            dictionary = Dictionary.from_files(
                to_path_dict(train_names).values(), lowercase=config.lowercase, max_size=config.vocab_size)
            dictionary.save(config.working_dir)
        self.dictionary = dictionary

        self.corpus = Corpus(
            to_path_dict(train_names + valid_names),
            maxlen=config.maxlen, preprocessor=Preprocessor(dictionary), lowercase=config.lowercase)
        self.ntokens = len(self.dictionary)
        logging.info("Vocabulary Size: {}".format(self.ntokens))

        eval_batch_size = 100
        self.test1_data = batchify(self.corpus.data['valid1'], eval_batch_size, shuffle=False)
        self.test2_data = batchify(self.corpus.data['valid2'], eval_batch_size, shuffle=False)
        self.train1_data = self.train2_data = None
        self.shuffle_training_data()

    def shuffle_training_data(self):
        self.train1_data = batchify(self.corpus.data['train1'], self.config.batch_size, shuffle=True)
        self.train2_data = batchify(self.corpus.data['train2'], self.config.batch_size, shuffle=True)
