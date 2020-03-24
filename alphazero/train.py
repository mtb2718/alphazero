from argparse import ArgumentParser
import os
import shutil
import time
import yaml

import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from alphazero.config import AlphaZeroConfig
from alphazero.model_server import ModelServer
from alphazero.replay_buffer import ReplayBufferDataset
from alphazero.selfplay import SelfPlayWorker
from alphazero.training import TrainingWorker


def load_config(logdir):
    # Load config file
    config_path = os.path.join(logdir, 'config.yaml')
    config = AlphaZeroConfig(config_path)
    return config


def init_logdir(args):
    # Create logdir and save copy of config and args for posterity
    os.makedirs(args.logdir)
    conf_path = os.path.join(args.logdir, 'config.yaml')
    shutil.copyfile(args.config, conf_path)
    with open(os.path.join(args.logdir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f)


def _init_training(logdir, device='cpu'):
    # Create summary writer for tensorboard logging
    summary_writer = SummaryWriter(log_dir=logdir)

    # Create dataset and model server instances
    config = load_config(logdir)
    dbpath = os.path.join(logdir, 'selfplay.sqlite')
    dataset = ReplayBufferDataset(config, dbpath)
    model_server = ModelServer(logdir)

    device = torch.device(device)
    train_worker = TrainingWorker(config, model_server, dataset, summary_writer, device)
    return train_worker


def _init_selfplay(logdir, device='cpu'):
    # Create dataset and model server instances
    config = load_config(logdir)
    dbpath = os.path.join(logdir, 'selfplay.sqlite')
    dataset = ReplayBufferDataset(config, dbpath)
    model_server = ModelServer(logdir)
    model_server.reset()

    device = torch.device(device)
    selfplay_worker = SelfPlayWorker(config, model_server, dataset, device)
    return selfplay_worker


def _selfplay(i, logdir, device):
    selfplay_worker = _init_selfplay(logdir, device)
    game = 1
    while True:
        game += 1
        selfplay_worker.play_game()


def _train(logdir, device):
    config = load_config(logdir)
    training_worker = _init_training(logdir, device)
    for _ in range(config.training['num_steps']):
        if not training_worker.process_batch():
            time.sleep(2)


def _synchronous_play_and_train(logdir, device):
    config = load_config(args.logdir)
    selfplay_worker = _init_selfplay(args.logdir, args.device)
    training_worker = _init_training(args.logdir, args.device)
    for _ in range(config.training['num_steps']):
        selfplay_worker.play_game()
        training_worker.process_batch()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-o', '--logdir',
        required=True,
        help="Directory for tensorboard logging and checkpoints.")
    parser.add_argument('-c', '--config',
        required=True,
        help="Path to training config yaml file.")
    parser.add_argument('-n', '--num-selfplay-workers',
        type=int,
        default=0,
        help='Number of subprocess selfplay workers (default 0 for no concurrency).')
    parser.add_argument('-d', '--device',
        default='cpu',
        help='The device to use for training and self-play inference, e.g. \'cuda:0\'. Currently only support for single device')
    args = parser.parse_args()
    init_logdir(args)

    # Start training
    # TODO: Properly break after num_steps training iterations
    # TODO: Kill workers when training is done
    if args.num_selfplay_workers == 0:
        _synchronous_play_and_train(args.logdir, args.device)
    else:
        # 1. Launch N self-play workers in new processes
        selfplay_ctx = mp.spawn(fn=_selfplay,
                                args=(args.logdir, args.device),
                                nprocs=args.num_selfplay_workers,
                                join=False)

        # 2. Train concurrently in main thread
        _train(args.logdir, args.device)

        # 3. Cleanup
        selfplay_ctx.join()

    print("Done.")
