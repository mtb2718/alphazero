import os

import torch
from torch.utils.tensorboard import SummaryWriter

from alphazero.config import AlphaZeroConfig
from alphazero.model_server import ModelServer
from alphazero.replay_buffer import ReplayBufferDataset
from alphazero.selfplay import SelfPlayWorker
from alphazero.training import TrainingWorker

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, 'configs/test_training.yaml')

def test_training(tmp_path):

    config = AlphaZeroConfig(CONFIG)
    model_server = ModelServer(tmp_path)
    model_server.reset()
    dataset = ReplayBufferDataset(config, os.path.join(tmp_path, 'dataset.sqlite'))
    summary_writer = SummaryWriter(tmp_path)
    device = torch.device('cpu')

    selfplay_worker = SelfPlayWorker(config, model_server, dataset, device)
    training_worker = TrainingWorker(config, model_server, dataset, summary_writer, device)

    for _ in range(config.training['num_steps']):
        selfplay_worker.play_game()

    for _ in range(config.training['num_steps']):
        assert training_worker.process_batch(), 'Should successfully proces batch'
