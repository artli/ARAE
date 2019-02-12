import bisect
import logging
import math
import os
import random
import time
import typing
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from data import Data
from models import Models
from utils import to_gpu, format_epoch, NamedTupleFromArgs


AutoencoderEvaluationResult = namedtuple('AutoencoderEvaluationResult', 'test_loss test_ppl test_accuracy')


def log_epoch_end(epoch, elapsed, autoencoder_results: typing.Iterable[AutoencoderEvaluationResult]):
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


class TrainingConfig(typing.NamedTuple, NamedTupleFromArgs):
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
