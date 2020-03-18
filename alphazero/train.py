from argparse import ArgumentParser
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from alphazero.config import AlphaZeroConfig
from alphazero.models import UniformModel
from alphazero.replay_buffer import ReplayBufferDataset, ReplayBufferSampler
from alphazero.selfplay import SelfPlayWorker


class ModelServer:

    def __init__(self, logdir, default_model=None, ver=0, ckpt_period=50):
        self._model = None
        self._default_model = default_model
        self._logdir = logdir
        self._model_ver = ver
        self._ckpt_period = ckpt_period

    def update(self, model):
        self._model = model # TODO: shared weights in memory?
        self._model_ver += 1
        if self._ckpt_period > 0 and self._model_ver % self._ckpt_period == 0:
            self.checkpoint()

    def latest(self):
        if self._model_ver > 0:
            return self._model_ver, self._model
        else:
            return 0, self._default_model

    def checkpoint(self):
        torch.save({
            'train_iter': self._model_ver,
            'model_state_dict': self._model.state_dict(),
            # TODO: Move checkpointing outside of this cache
            #'optimizer_state_dict': optimizer.state_dict(),
        }, f'{self._logdir}/ckpt.{self._model_ver}.pt')



class TrainWorker:
    def __init__(self, config, model_server, dataset, outdir, device):

        self._config = config
        self._model = config.Model().to(device)
        self._model_server = model_server
        self._loss = config.Loss()

        self._optimizer = SGD(self._model.parameters(),
                              lr=config.training['lr'],
                              momentum=config.training['momentum'],
                              weight_decay=config.training['weight_decay'])
        self._lr_schedule = MultiStepLR(self._optimizer,
                                        config.training['lr_schedule'],
                                        gamma=config.training['lr_schedule_gamma'])
        self._summary_writer = SummaryWriter(log_dir=outdir)

        self._dataset = dataset
        # TODO: Replay buffer settings from config
        self._sampler = ReplayBufferSampler(dataset,
                                            batch_size=config.training['batch_size'],
                                            bufferlen=(64 * 1024))
        self._dataloader = DataLoader(dataset,
                                      batch_sampler=self._sampler,
                                      pin_memory=True,
                                      num_workers=0)
        self._batch_iter = enumerate(self._dataloader)

    def __del__(self):
        self._summary_writer.close()

    def process_batch(self):
        train_iter, batch = next(self._batch_iter)

        # Don't update network unless we can make a full batch
        BATCH_SIZE = batch['x'].shape[0]
        N = len(self._dataset)
        if N < BATCH_SIZE:
            print(f"Skipping batch ({BATCH_SIZE}), too few examples in replay buffer ({N}).")
            return

        # Run inference, evaluate loss, backprop
        self._model.train()
        device = next(self._model.parameters()).device
        x = batch['x'].to(device)
        p = batch['p'].to(device)
        z = batch['z'].to(device)
        p_valid = batch['p_valid'].to(device)
        p_hat, v_hat = self._model(x, p_valid)
        prior_loss, value_loss = self._loss(p, z, p_hat, v_hat, p_valid)
        self._optimizer.zero_grad()
        total_loss = prior_loss + value_loss
        total_loss.backward()
        self._optimizer.step()
        self._lr_schedule.step()

        # Publish latest model
        self._model_server.update(self._model)

        # Log stats
        # TODO: Make summary logging a globally available.
        #       Then loss can return only a single value to backprop.
        self._summary_writer.add_scalar('total_loss', total_loss, train_iter)
        self._summary_writer.add_scalar('value_loss', value_loss, train_iter)
        self._summary_writer.add_scalar('prior_loss', prior_loss, train_iter)
        self._summary_writer.add_scalar('learning_rate', self._lr_schedule.get_lr()[0], train_iter)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-o', '--logdir',
        required=True,
        help="Directory for tensorboard logging and checkpoints.")
    parser.add_argument('-c', '--config',
        required=True,
        help="Path to training config yaml file.")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Using device', device)

    # Load config file and save copy to logdir
    os.makedirs(args.logdir)
    config = AlphaZeroConfig(args.config)
    config.save(os.path.join(args.logdir, 'config.yaml'))

    # Construct and configure self-play and training
    model_server = ModelServer(args.logdir, default_model=UniformModel())
    dataset = ReplayBufferDataset(config, os.path.join(args.logdir, 'selfplay.sqlite'))

    train_worker = TrainWorker(config, model_server, dataset, args.logdir, device)
    selfplay_worker = SelfPlayWorker(config, model_server, dataset)

    # Start training
    single_process = True
    num_selfplay = 1
    if single_process:
        done = False
        for _ in range(config.training['num_steps']):
            for _ in range(num_selfplay):
                selfplay_worker.play_game()
            train_worker.process_batch()
    else:
        # 1. Launch N self-play workers in new processes
        # 2. Launch training worker
        pass

    print("Done.")
