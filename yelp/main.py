import argparse
import glob
import json
import logging
import os
import sys
import traceback
from collections import Counter

from flask import Flask, request, jsonify

from data import Data, DataConfig, Dictionary, Preprocessor
from models import Models, ModelsConfig, TextOperations
from training import Trainer, TrainingConfig


class ArgumentParserWrapper:
    def __init__(self, *args, **kwargs):
        self._parser = argparse.ArgumentParser(*args, **kwargs)
        self._defaults = {}

    def add_argument(self, arg, default=None, **kwargs):
        if 'dest' in kwargs:
            name = kwargs['dest']
        else:
            name = arg.lstrip('-')
        self._defaults[name] = default
        return self._parser.add_argument(arg, **kwargs)

    def set_defaults(self, **kwargs):
        self._parser.set_defaults(**kwargs)

    def parse_args(self):
        args = self._parser.parse_args()
        explicitly_provided = {}
        for arg_name, default in self._defaults.items():
            if getattr(args, arg_name) is None:
                setattr(args, arg_name, default)
            else:
                explicitly_provided[arg_name] = getattr(args, arg_name)
        return args, explicitly_provided


def parse_args():
    parser = ArgumentParserWrapper(description='ARAE for Yelp transfer')

    # Path Arguments
    parser.add_argument('--data_dir', type=str, default='dataset', help='location of the data corpus')
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


def load_args() -> argparse.Namespace:
    args, explicit = parse_args()
    if not args.load_models:
        return args

    with open('{}/args.json'.format(args.working_dir)) as f:
        saved_arg_dict = json.load(f)
    saved_arg_dict.update(**explicit)
    saved_arg_dict['first_epoch'] = last_saved_epoch(args.working_dir) + 1

    saved_args = argparse.Namespace()
    saved_args.__dict__ = saved_arg_dict
    return saved_args


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
