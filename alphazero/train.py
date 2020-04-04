from argparse import ArgumentParser, Namespace
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
    config = AlphaZeroConfig(os.path.join(logdir, 'config.yaml'))
    with open(os.path.join(logdir, 'args.yaml')) as f:
        args = Namespace(**yaml.safe_load(f))
    return args, config


def init_logdir(args):
    # Create logdir and save copy of config and args for posterity
    os.makedirs(args.logdir)
    shutil.copyfile(args.config, os.path.join(args.logdir, 'config.yaml'))
    with open(os.path.join(args.logdir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f)


def _init_training(logdir, device, num_data_workers):
    # Create summary writer for tensorboard logging
    summary_writer = SummaryWriter(log_dir=logdir)

    # Create dataset and model server instances
    args, config = load_config(logdir)
    selfplay_db = args.selfplay_db or os.path.join(logdir, 'selfplay.sqlite')
    dataset = ReplayBufferDataset(config, selfplay_db)
    model_server = ModelServer(logdir)

    device = torch.device(device)
    train_worker = TrainingWorker(config, model_server, dataset, summary_writer, device, num_data_workers)
    return train_worker


def _init_selfplay(logdir, device):
    # Create dataset and model server instances
    args, config = load_config(logdir)
    dbpath = os.path.join(logdir, 'selfplay.sqlite')
    dataset = ReplayBufferDataset(config, dbpath)
    model_server = ModelServer(logdir)
    model_server.reset()

    device = torch.device(device)
    selfplay_worker = SelfPlayWorker(config, model_server, dataset, device)
    return selfplay_worker


def _selfplay(i, logdir, device):
    args, config = load_config(logdir)
    if args.selfplay_db is None:
        return
    selfplay_worker = _init_selfplay(logdir, device)
    while True:
        model_version = selfplay_worker.play_game()
        # Stop selfplay when training is done
        if model_version >= config.training['num_steps']:
            break


def _train(logdir, device, num_data_workers):
    args, config = load_config(logdir)
    training_worker = _init_training(logdir, device, num_data_workers)
    while True:
        new_model_ver = training_worker.process_batch()
        # Haven't yet started training--too few examples available
        if new_model_ver == 0:
            time.sleep(2)
        elif new_model_ver >= config.training['num_steps']:
            break


def _synchronous_play_and_train(logdir, device, num_data_workers):
    args, config = load_config(logdir)
    if args.selfplay_db is None:
        selfplay_worker = _init_selfplay(args.logdir, args.device)
    training_worker = _init_training(args.logdir, args.device, num_data_workers)
    while True:
        if args.selfplay_db is None:
            selfplay_worker.play_game()
        new_model_ver = training_worker.process_batch()
        if new_model_ver >= config.training['num_steps']:
            break


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-o', '--logdir',
        required=True,
        help="Directory for tensorboard logging and checkpoints.")
    parser.add_argument('-c', '--config',
        required=True,
        help="Path to training config yaml file.")
    parser.add_argument('--selfplay-db',
        default=None,
        help='Load training examples from game database instead of live self-play.')
    # TODO: DB Sampling strategies - exponential, curriculum
    parser.add_argument('-n', '--num-selfplay-workers',
        type=int,
        default=0,
        help='Number of subprocess selfplay workers (default 0 for no concurrency).')
    parser.add_argument('-w', '--num-data-workers',
        type=int,
        default=0,
        help='Number of worker processes to use in dataloader.')
    parser.add_argument('-d', '--device',
        default='cpu',
        help='The device to use for training and self-play inference, e.g. \'cuda:0\'. Currently only support for single device')
    args = parser.parse_args()

    init_logdir(args)

    # Start training
    if args.num_selfplay_workers == 0:
        _synchronous_play_and_train(args.logdir, args.device, args.num_data_workers)
    else:
        # 1. Launch N self-play workers in new processes
        selfplay_ctx = mp.spawn(fn=_selfplay,
                                args=(args.logdir, args.device),
                                nprocs=args.num_selfplay_workers,
                                join=False)

        # 2. Train concurrently in main thread
        _train(args.logdir, args.device, args.num_data_workers)

        # 3. Cleanup
        selfplay_ctx.join()

    print("Done.")
