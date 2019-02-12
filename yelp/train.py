import bisect
import logging
import json
import os
import sys
import glob
import random
import time
import typing
import math
import traceback
from argparse import ArgumentParser, Namespace
from collections import namedtuple, Counter
from typing import NamedTuple, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from flask import Flask, request, jsonify

from models import Seq2Seq2Decoder, MLP_D, MLP_G, MLP_Classify
from utils import to_gpu, Corpus, batchify, Preprocessor, Dictionary


def parse_args():
    parser = ArgumentParser(description='ARAE for Yelp transfer')

    # Path Arguments
    parser.add_argument('--data_dir', type=str, default='data', help='location of the data corpus')
    parser.add_argument('--working_dir', type=str, default='output', help='output directory name')
    parser.add_argument('--vocab_path', type=str, default="", help='path to load vocabulary from')
    parser.add_argument('--load_models', dest='load_models', action='store_true', help='load model weights from disk')
    parser.set_defaults(load_models=False)

    # Data Processing Arguments
    parser.add_argument('--vocab_size', type=int, default=30000,
                        help='cut vocabulary down to this size (most frequently seen words in train)')
    parser.add_argument('--maxlen', type=int, default=25, help='maximum sentence length')
    parser.add_argument('--lowercase', dest='lowercase', action='store_true', help='lowercase all text')
    parser.add_argument('--no_lowercase', dest='lowercase', action='store_false', help='not lowercase all text')
    parser.set_defaults(lowercase=True)

    # Model Arguments
    parser.add_argument('--emsize', type=int, default=128, help='size of word embeddings')
    parser.add_argument('--nhidden', type=int, default=128, help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=1, help='number of layers')
    parser.add_argument('--noise_r', type=float, default=0.1, help='stdev of noise for autoencoder (regularizer)')
    parser.add_argument('--noise_anneal', type=float, default=0.9995,
                        help='anneal noise_r exponentially by this every 100 iterations')
    parser.add_argument('--hidden_init', action='store_true', help="initialize decoder hidden state with encoder's")
    parser.add_argument('--arch_g', type=str, default='128-128', help='generator architecture (MLP)')
    parser.add_argument('--arch_d', type=str, default='128-128', help='critic/discriminator architecture (MLP)')
    parser.add_argument('--arch_classify', type=str, default='128-128', help='classifier architecture')
    parser.add_argument('--z_size', type=int, default=32, help='dimension of random noise z to feed into generator')
    parser.add_argument('--temp', type=float, default=1, help='softmax temperature (lower --> more discrete)')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout applied to layers (0 = no dropout)')

    # Training Arguments
    parser.add_argument('--epochs', type=int, default=25, help='maximum number of epochs')
    parser.add_argument('--first_epoch', type=int, default=1, help='first epoch number')
    parser.add_argument('--save_every', type=int, default=10, help='delete all model checkpoints except every nth')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='batch size')
    parser.add_argument('--niters_ae', type=int, default=1, help='number of autoencoder iterations in training')
    parser.add_argument('--niters_gan_d', type=int, default=5, help='number of discriminator iterations in training')
    parser.add_argument('--niters_gan_g', type=int, default=1, help='number of generator iterations in training')
    parser.add_argument('--niters_gan_ae', type=int, default=1, help='number of gan-into-ae iterations in training')
    parser.add_argument('--niters_gan_schedule', type=str, default='1',
                        help='epoch counts to increase number of GAN training iterations (increment by 1 each time)')
    parser.add_argument('--lr_ae', type=float, default=1, help='autoencoder learning rate')
    parser.add_argument('--lr_gan_g', type=float, default=1e-04, help='generator learning rate')
    parser.add_argument('--lr_gan_d', type=float, default=1e-04, help='critic/discriminator learning rate')
    parser.add_argument('--lr_classify', type=float, default=1e-04, help='classifier learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--clip', type=float, default=1, help='gradient clipping, max norm')
    parser.add_argument('--gan_gp_lambda', type=float, default=0.1, help='WGAN GP penalty lambda')
    parser.add_argument('--grad_lambda', type=float, default=0.01, help='WGAN into AE lambda')
    parser.add_argument('--classifier_lambda', type=float, default=1, help='lambda on classifier')

    # Evaluation Arguments
    parser.add_argument('--sample', action='store_true', help='sample when decoding for generation')
    parser.set_defaults(sample=False)
    parser.add_argument('--log_interval', type=int, default=200, help='interval to log autoencoder training results')

    # Other
    parser.add_argument('--mode', type=str, default='train',
                        help='what to do ("train", "transfer", "interpolate" or "serve")')
    parser.add_argument('--port', type=int, default=8002, help='port for the server')
    parser.add_argument('--midpoint_count', type=int, default=4, help='midpoint count for text interpolation')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--cuda', dest='cuda', action='store_true', help='use CUDA')
    parser.add_argument('--no_cuda', dest='cuda', action='store_false', help='not using CUDA')
    parser.set_defaults(cuda=True)
    parser.add_argument('--device_ids', type=str, default='')

    return parser.parse_args()


def last_saved_epoch(working_dir):
    paths = glob.glob(os.path.join(working_dir, 'models', '*_*.pt'))
    filenames = [os.path.split(path)[1] for path in paths]
    epochs = [int(filename.split('_')[0]) for filename in filenames]
    # Avoid accidentally selecting a partially-saved epoch
    last_epoch, _ = max(Counter(epochs).items(), key=lambda item: (item[1], item[0]))
    return last_epoch


def load_args() -> Namespace:
    args = parse_args()
    if not args.load_models:
        return args

    with open('{}/args.json'.format(args.working_dir)) as f:
        saved_arg_dict = json.load(f)
    saved_arg_dict.update(vars(args))
    saved_arg_dict['first_epoch'] = last_saved_epoch(args.working_dir) + 1

    saved_args = Namespace()
    saved_args.__dict__ = saved_arg_dict
    return saved_args


class MultipleInheritanceNamedTupleMeta(typing.NamedTupleMeta):
    # Makes extending NamedTuple possible
    # https://stackoverflow.com/questions/50367661/customizing-typing-namedtuple
    def __new__(mcls, typename, bases, ns):
        if NamedTuple in bases:
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


class ModelsConfig(NamedTuple, NamedTupleFromArgs):
    ntokens: int
    emsize: int
    nhidden: int
    nlayers: int
    noise_r: float
    hidden_init: bool
    dropout: float
    z_size: int
    arch_g: str
    arch_d: str
    arch_classify: str
    lr_ae: float
    lr_gan_g: float
    lr_gan_d: float
    lr_classify: float
    beta1: float
    cuda: bool


class Models:
    MODEL_NAMES = 'autoencoder', 'generator', 'discriminator', 'classifier'

    def __init__(self, config: ModelsConfig):
        self.config = config

        self.autoencoder = Seq2Seq2Decoder(
            emsize=config.emsize, nhidden=config.nhidden, ntokens=config.ntokens, nlayers=config.nlayers,
            noise_r=config.noise_r, hidden_init=config.hidden_init, dropout=config.dropout, gpu=config.cuda)
        self.generator = MLP_G(ninput=config.z_size, noutput=config.nhidden, layers=config.arch_g)
        self.discriminator = MLP_D(ninput=config.nhidden, noutput=1, layers=config.arch_d)
        self.classifier = MLP_Classify(ninput=config.nhidden, noutput=1, layers=config.arch_classify, gpu=config.cuda)

        self.autoencoder_opt = optim.SGD(
            self.autoencoder.parameters(), lr=config.lr_ae)
        self.generator_opt = optim.Adam(
            self.generator.parameters(), lr=config.lr_gan_g, betas=(config.beta1, 0.999))
        self.discriminator_opt = optim.Adam(
            self.discriminator.parameters(), lr=config.lr_gan_d, betas=(config.beta1, 0.999))
        self.classifier_opt = optim.Adam(
            self.classifier.parameters(), lr=config.lr_classify, betas=(config.beta1, 0.999))

        self.cross_entropy = nn.CrossEntropyLoss()

        if config.cuda:
            for attr_name in self.MODEL_NAMES + ('cross_entropy',):
                node = getattr(self, attr_name)
                setattr(self, attr_name, node.cuda())

    def _models_and_paths(self, working_dir, epoch, model_names=MODEL_NAMES):
        for model_name in model_names:
            model = getattr(self, model_name)
            path = os.path.join(working_dir, 'models', '{}_{}.pt'.format(format_epoch(epoch), model_name))
            yield model, path

    def save_state(self, working_dir, epoch):
        for model, path in self._models_and_paths(working_dir, epoch):
            if model is not None:
                with open(path, 'wb') as f:
                    torch.save(model.state_dict(), f)

    def cleanup(self, working_dir, epoch):
        for _, path in self._models_and_paths(working_dir, epoch):
            if os.path.exists(path):
                os.remove(path)

    def load_state(self, working_dir, epoch, model_names=MODEL_NAMES):
        for model, path in self._models_and_paths(working_dir, epoch, model_names=model_names):
            model.load_state_dict(
                torch.load(path, map_location=lambda storage, loc: to_gpu(self.config.cuda, storage)))
        for model_name in set(self.MODEL_NAMES) - set(model_names):
            setattr(self, model_name, None)


def format_epoch(epoch):
    if epoch is None:
        return 'final'
    else:
        return '{:03d}'.format(epoch)


AutoencoderEvaluationResult = namedtuple('AutoencoderEvaluationResult', 'test_loss test_ppl test_accuracy')


def log_epoch_end(epoch, elapsed, autoencoder_results: Iterable[AutoencoderEvaluationResult]):
    if epoch is None:
        description = 'Final evaluation'
    else:
        description = 'Epoch ' + format_epoch(epoch)
    logging.info('{} ended in {:5.2f} seconds'.format(description, elapsed))
    for number, result in enumerate(autoencoder_results, start=1):
        logging.info('Autoencoder {} evaluation:\ttest loss: {:5.2f},\ttest ppl {:5.2f},\tacc {:3.3f}'
                     .format(number, result.test_loss, result.test_ppl, result.test_accuracy))
    logging.info('-' * 89)


class TrainingContext:
    def __init__(self, first_epoch, final_epoch, gan_schedule, evaluation_noise, cuda):
        self.first_epoch = first_epoch
        self.final_epoch = final_epoch
        if not gan_schedule or gan_schedule[0] != 1:
            gan_schedule = [1] + gan_schedule
        self.gan_schedule = gan_schedule
        self.evaluation_noise = evaluation_noise

        self.ended = False
        self.epoch = None
        self.last_epoch_start = None
        self.one = to_gpu(cuda, torch.FloatTensor([1]))
        self.mone = self.one * -1

    def start(self):
        self.ended = False
        self.epoch = self.first_epoch
        self.last_epoch_start = time.time()
        logging.info('Starting training...')

    def next_epoch(self):
        if self.ended:
            raise ValueError('Training has ended')
        self.epoch += 1
        self.last_epoch_start = time.time()
        if self.epoch >= self.final_epoch:
            self.epoch = None
            self.ended = True

    def gan_iteration_count(self):
        return bisect.bisect_right(self.gan_schedule, self.epoch)


class TrainingConfig(NamedTuple, NamedTupleFromArgs):
    working_dir: str
    seed: int
    cuda: bool
    device_ids: str
    batch_size: int
    epochs: int
    first_epoch: int
    save_every: int
    log_interval: int
    niters_gan_schedule: str
    niters_ae: int
    niters_gan_g: int
    niters_gan_d: int
    niters_gan_ae: int
    classifier_lambda: float
    grad_lambda: float
    gan_gp_lambda: float
    temp: float
    sample: float
    clip: bool
    noise_anneal: float


class Trainer:
    def __init__(self, config: TrainingConfig, data: Data, models: Models):
        self.config = config
        self.data = data
        self.models = models

    def _seed(self):
        seed = self.config.seed
        # Set the random seed manually for reproducibility.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            if not self.config.cuda:
                logging.warning("You have a CUDA device, so you should probably run with --cuda")
            else:
                torch.cuda.manual_seed(seed)

    def create_context(self):
        gan_schedule = list(map(int, self.config.niters_gan_schedule.split("-")))
        evaluation_noise = to_gpu(self.config.cuda, Variable(
            torch.ones(self.config.batch_size, self.models.config.z_size)))
        evaluation_noise.data.normal_(0, 1)
        return TrainingContext(
            self.config.first_epoch, self.config.epochs, gan_schedule, evaluation_noise, self.config.cuda)

    def _on_epoch_end(self, context: TrainingContext, autoencoder_evaluation_datasets):
        autoencoder_results = []
        for number, dataset in enumerate(autoencoder_evaluation_datasets, start=1):
            test_loss, accuracy = self.evaluate_autoencoder(number, dataset, context.epoch)
            autoencoder_results.append(AutoencoderEvaluationResult(test_loss, np.exp(test_loss), accuracy))
        log_epoch_end(context.epoch, time.time() - context.last_epoch_start, autoencoder_results)

    def train(self, context: TrainingContext = None):
        if context is None:
            context = self.create_context()
        context.start()

        if self.config.device_ids:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.config.device_ids
        os.makedirs(self.config.working_dir, exist_ok=True)
        self._seed()

        while not context.ended:
            self.train_epoch(context)
            context.next_epoch()

        self._on_epoch_end(context, (self.data.test1_data, self.data.test2_data))

    def train_epoch(self, context: TrainingContext):
        total_loss_ae1 = 0
        total_loss_ae2 = 0
        classify_loss = 0
        start_time = time.time()
        niter = 0
        niter_global = 1

        # loop through all batches in training data
        while niter < len(self.data.train1_data) and niter < len(self.data.train2_data):
            # train autoencoder ----------------------------
            for i in range(self.config.niters_ae):
                if niter == len(self.data.train1_data):
                    break  # end of epoch
                total_loss_ae1, start_time = \
                    self.train_ae(1, self.data.train1_data[niter], total_loss_ae1, start_time, context.epoch, niter)
                total_loss_ae2, _ = \
                    self.train_ae(2, self.data.train2_data[niter], total_loss_ae2, start_time, context.epoch, niter)

                # train classifier ----------------------------
                classify_loss1, classify_acc1 = self.train_classifier(1, self.data.train1_data[niter])
                classify_loss2, classify_acc2 = self.train_classifier(2, self.data.train2_data[niter])
                classify_loss = (classify_loss1 + classify_loss2) / 2
                classify_acc = (classify_acc1 + classify_acc2) / 2
                # reverse to autoencoder
                self.classifier_regularize(1, self.data.train1_data[niter])
                self.classifier_regularize(2, self.data.train2_data[niter])

                niter += 1

            # train gan ----------------------------------
            for k in range(context.gan_iteration_count()):

                # train discriminator/critic
                for i in range(self.config.niters_gan_d):
                    # feed a seen sample within this epoch; good for early training
                    if i % 2 == 0:
                        batch = self.data.train1_data[random.randint(0, len(self.data.train1_data) - 1)]
                        whichdecoder = 1
                    else:
                        batch = self.data.train2_data[random.randint(0, len(self.data.train2_data) - 1)]
                        whichdecoder = 2
                    errD, errD_real, errD_fake = self.train_gan_d(context, whichdecoder, batch)

                # train generator
                for i in range(self.config.niters_gan_g):
                    errG = self.train_gan_g(context)

                # train autoencoder from d
                for i in range(self.config.niters_gan_ae):
                    if i % 2 == 0:
                        batch = self.data.train1_data[random.randint(0, len(self.data.train1_data) - 1)]
                        whichdecoder = 1
                    else:
                        batch = self.data.train2_data[random.randint(0, len(self.data.train2_data) - 1)]
                        whichdecoder = 2
                    errD_ = self.train_gan_d_into_ae(context, whichdecoder, batch)

            niter_global += 1
            if niter_global % 100 == 0:
                logging.info('[%d/%d][%d/%d] Loss_D: %.4f (Loss_D_real: %.4f '
                             'Loss_D_fake: %.4f) Loss_G: %.4f'
                             % (context.epoch, self.config.epochs, niter, len(self.data.train1_data),
                                 errD.item(), errD_real.item(),
                                 errD_fake.item(), errG.item()))
                logging.info("Classify loss: {:5.2f} | Classify accuracy: {:3.3f}\n".format(
                    classify_loss, classify_acc))

                # exponentially decaying noise on autoencoder
                self.models.autoencoder.noise_r = \
                    self.models.autoencoder.noise_r * self.config.noise_anneal

        self._on_epoch_end(context, (self.data.test1_data[:1000], self.data.test2_data[:1000]))

        self.evaluate_generator(1, context.evaluation_noise, context.epoch)
        self.evaluate_generator(2, context.evaluation_noise, context.epoch)

        logging.info('Saving models into ' + self.config.working_dir)
        self.models.save_state(self.config.working_dir, context.epoch)
        if (context.epoch - 1) % self.config.save_every != 0:
            self.models.cleanup(self.config.working_dir, context.epoch - 1)
        self.data.shuffle_training_data()

    def train_classifier(self, whichclass, batch):
        self.models.classifier.train()
        self.models.classifier.zero_grad()

        source, target, lengths = batch
        source = to_gpu(self.config.cuda, Variable(source))
        labels = to_gpu(self.config.cuda, Variable(torch.zeros(source.size(0)).fill_(whichclass - 1)))

        # Train
        code = self.models.autoencoder(0, source, lengths, noise=False, encode_only=True).detach()
        scores = self.models.classifier(code)
        classify_loss = F.binary_cross_entropy(scores.squeeze(1), labels)
        classify_loss.backward()
        self.models.classifier_opt.step()
        classify_loss = classify_loss.cpu().item()

        pred = scores.data.round().squeeze(1)
        accuracy = pred.eq(labels.data).float().mean()

        return classify_loss, accuracy

    def classifier_regularize(self, whichclass, batch):
        self.models.autoencoder.train()
        self.models.autoencoder.zero_grad()

        source, target, lengths = batch
        source = to_gpu(self.config.cuda, Variable(source))
        target = to_gpu(self.config.cuda, Variable(target))
        flippedclass = abs(2 - whichclass)
        labels = to_gpu(self.config.cuda, Variable(torch.zeros(source.size(0)).fill_(flippedclass)))

        # Train
        code = self.models.autoencoder(0, source, lengths, noise=False, encode_only=True)
        code.register_hook(lambda grad: grad * self.config.classifier_lambda)
        scores = self.models.classifier(code)
        classify_reg_loss = F.binary_cross_entropy(scores.squeeze(1), labels)
        classify_reg_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.models.autoencoder.parameters(), self.config.clip)
        self.models.autoencoder_opt.step()

        return classify_reg_loss

    def evaluate_autoencoder(self, whichdecoder, data_source, epoch):
        # Turn on evaluation mode which disables dropout.
        self.models.autoencoder.eval()
        total_loss = 0
        all_accuracies = 0
        bcnt = 0
        for i, batch in enumerate(data_source):
            source, target, lengths = batch
            with torch.no_grad():
                source = to_gpu(self.config.cuda, Variable(source))
                target = to_gpu(self.config.cuda, Variable(target))

            mask = target.gt(0)
            masked_target = target.masked_select(mask)
            # examples x ntokens
            output_mask = mask.unsqueeze(1).expand(mask.size(0), self.data.ntokens)

            hidden = self.models.autoencoder(0, source, lengths, noise=False, encode_only=True)

            # output: batch x seq_len x ntokens
            if whichdecoder == 1:
                output = self.models.autoencoder(1, source, lengths, noise=False)
                flattened_output = output.view(-1, self.data.ntokens)
                masked_output = \
                    flattened_output.masked_select(output_mask).view(-1, self.data.ntokens)
                # accuracy
                max_vals1, max_indices1 = torch.max(masked_output, 1)
                all_accuracies += \
                    torch.mean(max_indices1.eq(masked_target).float()).item()

                max_values1, max_indices1 = torch.max(output, 2)
                max_indices2 = self.models.autoencoder.generate(2, hidden, maxlen=50)
            else:
                output = self.models.autoencoder(2, source, lengths, noise=False)
                flattened_output = output.view(-1, self.data.ntokens)
                masked_output = \
                    flattened_output.masked_select(output_mask).view(-1, self.data.ntokens)
                # accuracy
                max_vals2, max_indices2 = torch.max(masked_output, 1)
                all_accuracies += \
                    torch.mean(max_indices2.eq(masked_target).float()).item()

                max_values2, max_indices2 = torch.max(output, 2)
                max_indices1 = self.models.autoencoder.generate(1, hidden, maxlen=50)

            total_loss += self.models.cross_entropy(masked_output / self.config.temp, masked_target).data
            bcnt += 1

            path_prefix = '{}/texts/{}_decoder{}'.format(
                self.config.working_dir, format_epoch(epoch), whichdecoder)
            decoder_source_path = path_prefix + '_source.txt'
            decoder_result_path = path_prefix + '_result.txt'
            with open(decoder_source_path, 'w') as f_from, open(decoder_result_path, 'w') as f_trans:
                max_indices1 = max_indices1.view(output.size(0), -1).data.cpu().numpy()
                max_indices2 = max_indices2.view(output.size(0), -1).data.cpu().numpy()
                target = target.view(output.size(0), -1).data.cpu().numpy()
                tran_indices = max_indices2 if whichdecoder == 1 else max_indices1
                for t, tran_idx in zip(target, tran_indices):
                    # real sentence
                    chars = " ".join([self.data.dictionary.idx2word[x] for x in t])
                    f_from.write(chars)
                    f_from.write("\n")
                    # transfer sentence
                    chars = " ".join([self.data.dictionary.idx2word[x] for x in tran_idx])
                    f_trans.write(chars)
                    f_trans.write("\n")

        return total_loss.item() / len(data_source), all_accuracies/bcnt

    def evaluate_generator(self, whichdecoder, noise, epoch):
        self.models.generator.eval()
        self.models.autoencoder.eval()

        # generate from fixed random noise
        fake_hidden = self.models.generator(noise)
        max_indices = self.models.autoencoder.generate(
            whichdecoder, fake_hidden, maxlen=50, sample=self.config.sample)

        with open('{}/texts/{}_generator{}.txt'.format(
                  self.config.working_dir, format_epoch(epoch), whichdecoder), "w") as f:
            max_indices = max_indices.data.cpu().numpy()
            for idx in max_indices:
                # generated sentence
                words = [self.data.dictionary.idx2word[x] for x in idx]
                # truncate sentences to first occurrence of <eos>
                truncated_sent = []
                for w in words:
                    if w != '<eos>':
                        truncated_sent.append(w)
                    else:
                        break
                chars = " ".join(truncated_sent)
                f.write(chars)
                f.write("\n")

    def train_ae(self, whichdecoder, batch, total_loss_ae, start_time, epoch, i):
        self.models.autoencoder.train()
        self.models.autoencoder_opt.zero_grad()

        source, target, lengths = batch
        source = to_gpu(self.config.cuda, Variable(source))
        target = to_gpu(self.config.cuda, Variable(target))

        mask = target.gt(0)
        masked_target = target.masked_select(mask)
        output_mask = mask.unsqueeze(1).expand(mask.size(0), self.data.ntokens)
        output = self.models.autoencoder(whichdecoder, source, lengths, noise=True)
        flat_output = output.view(-1, self.data.ntokens)
        masked_output = flat_output.masked_select(output_mask).view(-1, self.data.ntokens)
        loss = self.models.cross_entropy(masked_output / self.config.temp, masked_target)
        loss.backward()

        # `clip_grad_norm_` to prevent exploding gradient in RNNs / LSTMs
        torch.nn.utils.clip_grad_norm_(self.models.autoencoder.parameters(), self.config.clip)
        self.models.autoencoder_opt.step()

        total_loss_ae += loss.data

        if i % self.config.log_interval == 0 and i > 0:
            probs = F.softmax(masked_output, dim=-1)
            max_vals, max_indices = torch.max(probs, 1)
            accuracy = torch.mean(max_indices.eq(masked_target).float()).item()
            cur_loss = total_loss_ae.item() / self.config.log_interval
            elapsed = time.time() - start_time
            logging.info('| epoch {} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                         'loss {:5.2f} | ppl {:8.2f} | acc {:8.2f}'
                         .format(format_epoch(epoch), i, len(self.data.train1_data),
                                 elapsed * 1000 / self.config.log_interval,
                                 cur_loss, math.exp(cur_loss), accuracy))

            total_loss_ae = 0
            start_time = time.time()

        return total_loss_ae, start_time

    def train_gan_g(self, context: TrainingContext):
        self.models.generator.train()
        self.models.generator.zero_grad()

        noise = to_gpu(self.config.cuda,
                       Variable(torch.ones(self.config.batch_size, self.models.config.z_size)))
        noise.data.normal_(0, 1)
        fake_hidden = self.models.generator(noise)
        errG = self.models.discriminator(fake_hidden)
        errG.backward(context.one)
        self.models.generator_opt.step()

        return errG

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        """ Stolen from https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py """
        bsz = real_data.size(0)
        alpha = torch.rand(bsz, 1)
        alpha = alpha.expand(bsz, real_data.size(1))  # only works for 2D
        alpha = to_gpu(self.config.cuda, alpha)
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = Variable(interpolates, requires_grad=True)
        disc_interpolates = netD(interpolates)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=to_gpu(self.config.cuda, torch.ones(disc_interpolates.size())),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.config.gan_gp_lambda
        return gradient_penalty

    def train_gan_d(self, context: TrainingContext, whichdecoder, batch):
        self.models.discriminator.train()
        self.models.discriminator_opt.zero_grad()

        # positive samples ----------------------------
        # generate real codes
        source, target, lengths = batch
        source = to_gpu(self.config.cuda, Variable(source))
        target = to_gpu(self.config.cuda, Variable(target))

        # batch_size x nhidden
        real_hidden = self.models.autoencoder(whichdecoder, source, lengths, noise=False, encode_only=True)

        # loss / backprop
        errD_real = self.models.discriminator(real_hidden)
        errD_real.backward(context.one)

        # negative samples ----------------------------
        # generate fake codes
        noise = to_gpu(self.config.cuda, Variable(torch.ones(source.size(0), self.models.config.z_size)))
        noise.data.normal_(0, 1)

        # loss / backprop
        fake_hidden = self.models.generator(noise)
        errD_fake = self.models.discriminator(fake_hidden.detach())
        errD_fake.backward(context.mone)

        # gradient penalty
        gradient_penalty = self.calc_gradient_penalty(self.models.discriminator, real_hidden.data, fake_hidden.data)
        gradient_penalty.backward()

        self.models.discriminator_opt.step()
        errD = -(errD_real - errD_fake)

        return errD, errD_real, errD_fake

    def train_gan_d_into_ae(self, context: TrainingContext, whichdecoder, batch):
        self.models.autoencoder.train()
        self.models.autoencoder_opt.zero_grad()

        source, target, lengths = batch
        source = to_gpu(self.config.cuda, Variable(source))
        target = to_gpu(self.config.cuda, Variable(target))
        real_hidden = self.models.autoencoder(whichdecoder, source, lengths, noise=False, encode_only=True)
        real_hidden.register_hook(lambda grad: grad * self.config.grad_lambda)
        errD_real = self.models.discriminator(real_hidden)
        errD_real.backward(context.mone)
        torch.nn.utils.clip_grad_norm_(self.models.autoencoder.parameters(), self.config.clip)

        self.models.autoencoder_opt.step()

        return errD_real


def get_answer(prompt):
    answer = input(prompt).lower()
    if answer in ('y', 'yes'):
        return True
    elif answer in ('n', 'no'):
        return False
    else:
        return None


def set_up_working_dir(working_dir):
    patterns = 'texts/*.txt', 'models/*.pt', '*.json'
    conflicting_paths = [
        path for pattern in patterns
        for path in glob.glob(os.path.join(working_dir, pattern))]
    if conflicting_paths:
        answer = None
        while answer is None:
            answer = get_answer(
                'There are training-related files in the working directory ({}). Use the --load_models flag to'
                ' reuse them. Are you sure you want to delete them and start from scratch? (y/n) '.format(working_dir))
        if not answer:
            print('Aborting due to non-empty working directory...', file=sys.stderr)
            sys.exit(1)
        for path in conflicting_paths:
            os.remove(path)
    os.makedirs(os.path.join(working_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(working_dir, 'texts'), exist_ok=True)


def set_up_logging(working_dir):
    format_string = '%(asctime)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(format_string)

    file_handler = logging.FileHandler('{}/log.txt'.format(working_dir))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logging.basicConfig(level=logging.INFO, format=format_string)
    logging.getLogger('').addHandler(file_handler)


class TextOperations:
    def __init__(self, preprocessor: Preprocessor, models: Models, cuda: bool, maxlen: int):
        self.preprocessor = preprocessor
        self.models = models
        self.cuda = cuda
        self.maxlen = maxlen

    def transfer_batch(self, batch, style_code, temp=None):
        source, _, lengths = batch
        with torch.no_grad():
            source = to_gpu(self.cuda, Variable(source))
            hidden = self.models.autoencoder(0, source, lengths, noise=False, encode_only=True)
            output = self.models.autoencoder.generate(
                style_code, hidden, maxlen=50, sample=temp is not None, temp=temp)
            output = output.view(len(source), -1).data.cpu().numpy()
        return output

    def interpolate_batches(self, batch1, batch2, midpoint_count, style_code, temp):
        source1, _, lengths1 = batch1
        source2, _, lengths2 = batch2
        coeffs = np.linspace(0, 1, midpoint_count + 2)
        results = []
        with torch.no_grad():
            source1 = to_gpu(self.cuda, Variable(source1))
            source2 = to_gpu(self.cuda, Variable(source2))
            hidden1 = self.models.autoencoder(0, source1, lengths1, noise=False, encode_only=True)
            hidden2 = self.models.autoencoder(0, source2, lengths2, noise=False, encode_only=True)
            points = [hidden2 * coeff + hidden1 * (1 - coeff) for coeff in coeffs]
            for point in points:
                output = self.models.autoencoder.generate(
                    style_code, point, maxlen=50, sample=temp is not None, temp=temp)
                output = output.view(len(source1), -1).data.cpu().numpy()
                results.append(output)
        return results

    def transfer_text(self, text, style_code, temp=None):
        batch = self.preprocessor.text_to_batch(text, maxlen=self.maxlen)
        transferred = self.transfer_batch(batch, style_code, temp)
        return self.preprocessor.batch_to_text(transferred)

    def interpolate_texts(self, text1, text2, midpoint_count, style_code, temp=None):
        batch1 = self.preprocessor.text_to_batch(text1, maxlen=self.maxlen)
        batch2 = self.preprocessor.text_to_batch(text2, maxlen=self.maxlen)
        if len(batch1) != len(batch2):
            raise ValueError('Can only interpolate between texts with the same number of sentences')
        results = self.interpolate_batches(batch1, batch2, midpoint_count, style_code, temp)
        return list(map(self.preprocessor.batch_to_text, results))


class Server:
    def __init__(self, operations: TextOperations, app: Flask, port: int, default_midpoint_count):
        self._operations = operations
        self._app = app
        self._port = port
        self._default_midpoint_count = default_midpoint_count
        app.route('/api/transfer', methods=['GET', 'POST'])(self.transfer)
        app.route('/api/interpolate', methods=['GET', 'POST'])(self.interpolate)

    @staticmethod
    def _parameters():
        return request.args if request.method == 'GET' else request.form

    def transfer(self):
        parameters = self._parameters()
        text = parameters['text']
        temperature = parameters.get('temperature')
        if temperature:
            temperature = float(temperature)
        positive = self._operations.transfer_text(text, 1, temperature)
        negative = self._operations.transfer_text(text, 2, temperature)
        return jsonify(dict(positive=positive, negative=negative))

    def interpolate(self):
        parameters = self._parameters()
        text1 = parameters['text1']
        text2 = parameters['text2']
        midpoint_count = parameters.get('midpoint_count', self._default_midpoint_count)
        midpoint_count = int(midpoint_count)
        temperature = parameters.get('temperature')
        if temperature:
            temperature = float(temperature)
        positive = self._operations.interpolate_texts(text1, text2, midpoint_count, 1, temperature)
        negative = self._operations.interpolate_texts(text1, text2, midpoint_count, 2, temperature)
        return jsonify(dict(positive=positive, negative=negative))

    def serve(self):
        self._app.run(host='0.0.0.0', port=self._port, debug=True, threaded=True)


def main():
    args = load_args()
    if args.mode != 'train' and not args.load_models:
        print('--load_models is required in all modes except "train"', file=sys.stderr)
        sys.exit(1)
    if not args.load_models:
        set_up_working_dir(args.working_dir)
    set_up_logging(args.working_dir)
    logging.info('train.py launched with args: ' + str(vars(args)))

    data = None
    if args.load_models:
        dictionary = Dictionary.load(args.working_dir)
    else:
        data = Data(DataConfig.from_args(args))
        dictionary = data.dictionary
    preprocessor = Preprocessor(dictionary)

    models = Models(ModelsConfig.from_args(args, ntokens=len(dictionary)))
    logging.info("Models successfully built:")
    for model_name in models.MODEL_NAMES:
        logging.info(getattr(models, model_name))
    if args.load_models:
        model_names = Models.MODEL_NAMES if args.mode == 'train' else ('autoencoder',)
        models.load_state(args.working_dir, args.first_epoch - 1, model_names=model_names)
        logging.info('Model weights successfully loaded from ' + args.working_dir)

    with open('{}/args.json'.format(args.working_dir), 'w') as f:
        json.dump(vars(args), f)
    logging.info('Saved the current arguments into {}/args.json'.format(args.working_dir))

    operations = TextOperations(preprocessor, models, args.cuda, args.maxlen)

    if args.mode == 'train':
        if data is None:
            data = Data(DataConfig.from_args(args), dictionary)
        logging.info("Data successfully loaded")
        trainer = Trainer(TrainingConfig.from_args(args), data, models)
        trainer.train()
    elif args.mode == 'transfer':
        while True:
            text = input('> ')
            if not text:
                continue
            for style_code in (1, 2):
                transferred = operations.transfer_text(text, style_code, args.temp if args.sample else None)
                print('Transferred to style {}:\t{}'.format(style_code, transferred))
    elif args.mode == 'interpolate':
        while True:
            text1 = input('1> ')
            text2 = input('2> ')
            if not text1 or not text2:
                continue
            for style_code in (1, 2):
                try:
                    results = operations.interpolate_texts(
                        text1, text2, args.midpoint_count, style_code, args.temp if args.sample else None)
                except ValueError:
                    traceback.print_exc()
                    break
                print(' Interpolation in style {}:'.format(style_code))
                print('\n'.join(results))
                print()
    elif args.mode == 'serve':
        app = Flask(__name__)
        server = Server(operations, app, args.port, args.midpoint_count)
        server.serve()
    else:
        logging.error('"{}" is an unrecognized option for "mode". Nothing to do, exiting...'.format(args.mode))


if __name__ == '__main__':
    main()
