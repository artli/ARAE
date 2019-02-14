import os
from functools import partial
from typing import NamedTuple

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from data import Preprocessor
from utils import to_gpu, format_epoch, NamedTupleFromArgs, batchify


class MLP_Classify(nn.Module):
    def __init__(self, ninput, noutput, layers, activation=nn.ReLU(), gpu=False):
        super(MLP_Classify, self).__init__()
        self.ninput = ninput
        self.noutput = noutput
        self.gpu = gpu

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes) - 1):
            layer = to_gpu(self.gpu, nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            self.layers.append(layer)
            self.add_module("layer" + str(i + 1), layer)

            # No batch normalization in first layer
            if i != 0:
                bn = to_gpu(self.gpu, nn.BatchNorm1d(layer_sizes[i + 1]))
                self.layers.append(bn)
                self.add_module("bn" + str(i + 1), bn)

            self.layers.append(to_gpu(self.gpu, activation))
            self.add_module("activation" + str(i + 1), activation)

        layer = to_gpu(self.gpu, nn.Linear(layer_sizes[-1], noutput))
        self.layers.append(layer)
        self.add_module("layer" + str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = F.sigmoid(x)
        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass


class Seq2Seq2Decoder(nn.Module):
    def __init__(self, emsize, nhidden, ntokens, nlayers, noise_r=0.2,
                 share_decoder_emb=False, hidden_init=False, dropout=0, gpu=False):
        super(Seq2Seq2Decoder, self).__init__()
        self.nhidden = nhidden
        self.emsize = emsize
        self.ntokens = ntokens
        self.nlayers = nlayers
        self.noise_r = noise_r
        self.hidden_init = hidden_init
        self.dropout = dropout
        self.gpu = gpu

        self.start_symbols = to_gpu(gpu, Variable(torch.ones(10, 1).long()))

        # Vocabulary embedding
        self.embedding = nn.Embedding(ntokens, emsize)
        self.embedding_decoder1 = nn.Embedding(ntokens, emsize)
        self.embedding_decoder2 = nn.Embedding(ntokens, emsize)

        # RNN Encoder and Decoder
        self.encoder = nn.LSTM(input_size=emsize,
                               hidden_size=nhidden,
                               num_layers=nlayers,
                               dropout=dropout,
                               batch_first=True)

        decoder_input_size = emsize + nhidden
        self.decoder1 = nn.LSTM(input_size=decoder_input_size,
                                hidden_size=nhidden,
                                num_layers=1,
                                dropout=dropout,
                                batch_first=True)
        self.decoder2 = nn.LSTM(input_size=decoder_input_size,
                                hidden_size=nhidden,
                                num_layers=1,
                                dropout=dropout,
                                batch_first=True)

        # Initialize Linear Transformation
        self.linear = nn.Linear(nhidden, ntokens)

        self.init_weights()

        if share_decoder_emb:
            self.embedding_decoder2.weight = self.embedding_decoder1.weight

    def init_weights(self):
        initrange = 0.1

        # Initialize Vocabulary Matrix Weight
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding_decoder1.weight.data.uniform_(-initrange, initrange)
        self.embedding_decoder2.weight.data.uniform_(-initrange, initrange)

        # Initialize Encoder and Decoder Weights
        for p in self.encoder.parameters():
            p.data.uniform_(-initrange, initrange)
        for p in self.decoder1.parameters():
            p.data.uniform_(-initrange, initrange)
        for p in self.decoder2.parameters():
            p.data.uniform_(-initrange, initrange)

        # Initialize Linear Weight
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)

    def init_hidden(self, bsz):
        zeros1 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        zeros2 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        return (to_gpu(self.gpu, zeros1), to_gpu(self.gpu, zeros2))

    def init_state(self, bsz):
        zeros = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        return to_gpu(self.gpu, zeros)

    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

    def forward(self, whichdecoder, indices, lengths, noise=False, encode_only=False):
        batch_size, maxlen = indices.size()

        hidden = self.encode(indices, lengths, noise)

        if hidden.requires_grad:
            hidden.register_hook(self.store_grad_norm)

        if encode_only:
            return hidden

        decoded = self.decode(whichdecoder, hidden, batch_size, maxlen,
                              indices=indices, lengths=lengths)

        return decoded

    def encode(self, indices, lengths, noise):
        embeddings = self.embedding(indices)
        packed_embeddings = pack_padded_sequence(input=embeddings,
                                                 lengths=lengths,
                                                 batch_first=True)

        # Encode
        packed_output, state = self.encoder(packed_embeddings)
        hidden, cell = state

        # batch_size x nhidden
        hidden = hidden[-1]  # get hidden state of last layer of encoder

        # normalize to unit ball (l2 norm of 1) - p=2, dim=1
        norms = torch.norm(hidden, 2, 1)

        # For older versions of PyTorch use:
        hidden = torch.div(hidden, norms.unsqueeze(1).expand_as(hidden))
        # For newest version of PyTorch (as of 8/25) use this:
        # hidden = torch.div(hidden, norms.unsqueeze(1).expand_as(hidden))

        if noise and self.noise_r > 0:
            gauss_noise = torch.normal(mean=torch.zeros(hidden.size()),
                                       std=self.noise_r)
            hidden = hidden + to_gpu(self.gpu, Variable(gauss_noise))

        return hidden

    def decode(self, whichdecoder, hidden, batch_size, maxlen, indices=None, lengths=None):
        # batch x hidden
        all_hidden = hidden.unsqueeze(1).repeat(1, maxlen, 1)

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state = (hidden.unsqueeze(0), self.init_state(batch_size))
        else:
            state = self.init_hidden(batch_size)

        if whichdecoder == 1:
            embeddings = self.embedding_decoder1(indices)
        else:
            embeddings = self.embedding_decoder2(indices)

        augmented_embeddings = torch.cat([embeddings, all_hidden], 2)
        packed_embeddings = pack_padded_sequence(input=augmented_embeddings,
                                                 lengths=lengths,
                                                 batch_first=True)

        if whichdecoder == 1:
            packed_output, state = self.decoder1(packed_embeddings, state)
        else:
            packed_output, state = self.decoder2(packed_embeddings, state)
        output, lengths = pad_packed_sequence(packed_output, batch_first=True)

        # reshape to batch_size*maxlen x nhidden before linear over vocab
        decoded = self.linear(output.contiguous().view(-1, self.nhidden))
        decoded = decoded.view(batch_size, maxlen, self.ntokens)

        return decoded

    def generate(self, whichdecoder, hidden, maxlen, sample=False, temp=1.0):
        """Generate through decoder; no backprop"""

        batch_size = hidden.size(0)

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state = (hidden.unsqueeze(0), self.init_state(batch_size))
        else:
            state = self.init_hidden(batch_size)

        # <sos>
        self.start_symbols.data.resize_(batch_size, 1)
        self.start_symbols.data.fill_(1)
        self.start_symbols = to_gpu(self.gpu, self.start_symbols)

        if whichdecoder == 1:
            embedding = self.embedding_decoder1(self.start_symbols)
        else:
            embedding = self.embedding_decoder2(self.start_symbols)

        inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

        # unroll
        all_indices = []
        for i in range(maxlen):
            if whichdecoder == 1:
                output, state = self.decoder1(inputs, state)
            else:
                output, state = self.decoder2(inputs, state)
            overvocab = self.linear(output.squeeze(1))

            if not sample:
                vals, indices = torch.max(overvocab, 1)
                indices = indices.unsqueeze(1)
            else:
                probs = F.softmax(overvocab / temp)
                indices = torch.multinomial(probs, 1)

            all_indices.append(indices)

            if whichdecoder == 1:
                embedding = self.embedding_decoder1(indices)
            else:
                embedding = self.embedding_decoder2(indices)
            inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

        max_indices = torch.cat(all_indices, 1)

        return max_indices


class MLP_D(nn.Module):
    def __init__(self, ninput, noutput, layers,
                 activation=nn.LeakyReLU(0.2), gpu=False):
        super(MLP_D, self).__init__()
        self.ninput = ninput
        self.noutput = noutput

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            self.layers.append(layer)
            self.add_module("layer" + str(i + 1), layer)

            # No batch normalization after first layer
            if i != 0:
                bn = nn.BatchNorm1d(layer_sizes[i + 1], eps=1e-05, momentum=0.1)
                self.layers.append(bn)
                self.add_module("bn" + str(i + 1), bn)

            self.layers.append(activation)
            self.add_module("activation" + str(i + 1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer" + str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = torch.mean(x)
        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass


class MLP_G(nn.Module):
    def __init__(self, ninput, noutput, layers, activation=nn.ReLU()):
        super(MLP_G, self).__init__()
        self.ninput = ninput
        self.noutput = noutput

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            self.layers.append(layer)
            self.add_module("layer" + str(i + 1), layer)

            bn = nn.BatchNorm1d(layer_sizes[i + 1], eps=1e-05, momentum=0.1)
            self.layers.append(bn)
            self.add_module("bn" + str(i + 1), bn)

            self.layers.append(activation)
            self.add_module("activation" + str(i + 1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer" + str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass


class Seq2Seq(nn.Module):
    def __init__(self, emsize, nhidden, ntokens, nlayers, noise_r=0.2,
                 hidden_init=False, dropout=0, gpu=False):
        super(Seq2Seq, self).__init__()
        self.nhidden = nhidden
        self.emsize = emsize
        self.ntokens = ntokens
        self.nlayers = nlayers
        self.noise_r = noise_r
        self.hidden_init = hidden_init
        self.dropout = dropout
        self.gpu = gpu

        self.start_symbols = to_gpu(gpu, Variable(torch.ones(10, 1).long()))

        # Vocabulary embedding
        self.embedding = nn.Embedding(ntokens, emsize)
        self.embedding_decoder = nn.Embedding(ntokens, emsize)

        # RNN Encoder and Decoder
        self.encoder = nn.LSTM(input_size=emsize,
                               hidden_size=nhidden,
                               num_layers=nlayers,
                               dropout=dropout,
                               batch_first=True)

        decoder_input_size = emsize + nhidden
        self.decoder = nn.LSTM(input_size=decoder_input_size,
                               hidden_size=nhidden,
                               num_layers=1,
                               dropout=dropout,
                               batch_first=True)

        # Initialize Linear Transformation
        self.linear = nn.Linear(nhidden, ntokens)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1

        # Initialize Vocabulary Matrix Weight
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding_decoder.weight.data.uniform_(-initrange, initrange)

        # Initialize Encoder and Decoder Weights
        for p in self.encoder.parameters():
            p.data.uniform_(-initrange, initrange)
        for p in self.decoder.parameters():
            p.data.uniform_(-initrange, initrange)

        # Initialize Linear Weight
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)

    def init_hidden(self, bsz):
        zeros1 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        zeros2 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        return (to_gpu(self.gpu, zeros1), to_gpu(self.gpu, zeros2))

    def init_state(self, bsz):
        zeros = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        return to_gpu(self.gpu, zeros)

    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

    def forward(self, indices, lengths, noise, encode_only=False):
        batch_size, maxlen = indices.size()

        hidden = self.encode(indices, lengths, noise)

        if encode_only:
            return hidden

        if hidden.requires_grad:
            hidden.register_hook(self.store_grad_norm)

        decoded = self.decode(hidden, batch_size, maxlen,
                              indices=indices, lengths=lengths)

        return decoded

    def encode(self, indices, lengths, noise):
        embeddings = self.embedding(indices)
        packed_embeddings = pack_padded_sequence(input=embeddings,
                                                 lengths=lengths,
                                                 batch_first=True)

        # Encode
        packed_output, state = self.encoder(packed_embeddings)

        hidden, cell = state
        # batch_size x nhidden
        hidden = hidden[-1]  # get hidden state of last layer of encoder

        # normalize to unit ball (l2 norm of 1) - p=2, dim=1
        norms = torch.norm(hidden, 2, 1)

        # For older versions of PyTorch use:
        hidden = torch.div(hidden, norms.expand_as(hidden))
        # For newest version of PyTorch (as of 8/25) use this:
        # hidden = torch.div(hidden, norms.unsqueeze(1).expand_as(hidden))

        if noise and self.noise_r > 0:
            gauss_noise = torch.normal(mean=torch.zeros(hidden.size()),
                                       std=self.noise_r)
            hidden = hidden + to_gpu(self.gpu, Variable(gauss_noise))

        return hidden

    def decode(self, hidden, batch_size, maxlen, indices=None, lengths=None):
        # batch x hidden
        all_hidden = hidden.unsqueeze(1).repeat(1, maxlen, 1)

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state = (hidden.unsqueeze(0), self.init_state(batch_size))
        else:
            state = self.init_hidden(batch_size)

        embeddings = self.embedding_decoder(indices)
        augmented_embeddings = torch.cat([embeddings, all_hidden], 2)
        packed_embeddings = pack_padded_sequence(input=augmented_embeddings,
                                                 lengths=lengths,
                                                 batch_first=True)

        packed_output, state = self.decoder(packed_embeddings, state)
        output, lengths = pad_packed_sequence(packed_output, batch_first=True)

        # reshape to batch_size*maxlen x nhidden before linear over vocab
        decoded = self.linear(output.contiguous().view(-1, self.nhidden))
        decoded = decoded.view(batch_size, maxlen, self.ntokens)

        return decoded

    def generate(self, hidden, maxlen, sample=False, temp=1.0):
        """Generate through decoder; no backprop"""

        batch_size = hidden.size(0)

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state = (hidden.unsqueeze(0), self.init_state(batch_size))
        else:
            state = self.init_hidden(batch_size)

        # <sos>
        self.start_symbols.data.resize_(batch_size, 1)
        self.start_symbols.data.fill_(1)

        embedding = self.embedding_decoder(self.start_symbols)
        inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

        # unroll
        all_indices = []
        for i in range(maxlen):
            output, state = self.decoder(inputs, state)
            overvocab = self.linear(output.squeeze(1))

            if not sample:
                vals, indices = torch.max(overvocab, 1)
            else:
                # sampling
                probs = F.softmax(overvocab / temp)
                indices = torch.multinomial(probs, 1)

            all_indices.append(indices)

            embedding = self.embedding_decoder(indices)
            inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

        max_indices = torch.cat(all_indices, 1)

        return max_indices


def generate(autoencoder, gan_gen, z, vocab, sample, maxlen):
    """
    Assume noise is batch_size x z_size
    """
    if type(z) == Variable:
        noise = z
    elif type(z) == torch.FloatTensor or type(z) == torch.cuda.FloatTensor:
        with torch.no_grad():
            noise = Variable(z)
    elif type(z) == np.ndarray:
        with torch.no_grad():
            noise = Variable(torch.from_numpy(z).float())
    else:
        raise ValueError("Unsupported input type (noise): {}".format(type(z)))

    gan_gen.eval()
    autoencoder.eval()

    # generate from random noise
    fake_hidden = gan_gen(noise)
    max_indices = autoencoder.generate(hidden=fake_hidden,
                                       maxlen=maxlen,
                                       sample=sample)

    max_indices = max_indices.data.cpu().numpy()
    sentences = []
    for idx in max_indices:
        # generated sentence
        words = [vocab[x] for x in idx]
        # truncate sentences to first occurrence of <eos>
        truncated_sent = []
        for w in words:
            if w != '<eos>':
                truncated_sent.append(w)
            else:
                break
        sent = " ".join(truncated_sent)
        sentences.append(sent)

    return sentences


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


class TextOperations:
    def __init__(self, preprocessor: Preprocessor, models: Models, cuda: bool, maxlen: int):
        self.preprocessor = preprocessor
        self.models = models
        self.cuda = cuda
        self.maxlen = maxlen

    def transfer_batch(self, batch, style_code, temp=None):
        with torch.no_grad():
            source = to_gpu(self.cuda, Variable(batch.source))
            hidden = self.models.autoencoder(0, source, batch.lengths, noise=False, encode_only=True)
            output = self.models.autoencoder.generate(
                style_code, hidden, maxlen=50, sample=temp is not None, temp=temp)
            output = output.view(len(source), -1).data.cpu().numpy()
        return output

    def interpolate_batches(self, batch1, batch2, midpoint_count, style_code, temp):
        source1, _, lengths1, _ = batch1
        source2, _, lengths2, _ = batch2
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
                    style_code, point, maxlen=self.maxlen, sample=temp is not None, temp=temp)
                output = output.view(len(source1), -1).data.cpu().numpy()
                results.append(output)
        return results

    def transfer_text(self, text, style_code, temp=None):
        batch = self.preprocessor.text_to_training_batch(text, maxlen=self.maxlen)
        transferred = self.transfer_batch(batch, style_code, temp)
        return self.preprocessor.model_batch_to_text(transferred, inverse_permutation=batch.inverse_permutation)

    def interpolate_texts(self, text1, text2, midpoint_count, style_code, temp=None):
        sentences1 = self.preprocessor.text_to_sentences(text1)
        sentences2 = self.preprocessor.text_to_sentences(text2)
        if len(sentences1) != len(sentences2):
            raise ValueError('Can only interpolate between texts with the same number of sentences')

        sentence_results = []
        for sentence1, sentence2 in zip(sentences1, sentences2):
            batch1 = batchify([self.preprocessor.sentence_to_ids(sentence1, maxlen=self.maxlen)], 1)[0]
            batch2 = batchify([self.preprocessor.sentence_to_ids(sentence2, maxlen=self.maxlen)], 1)[0]
            interpolates = self.interpolate_batches(batch1, batch2, midpoint_count, style_code, temp)
            sentence_results.append([batch[0] for batch in interpolates])
        return list(map(self.preprocessor.model_batch_to_text, zip(*sentence_results)))
