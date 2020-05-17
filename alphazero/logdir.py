import os

import torch

from alphazero.config import AlphaZeroConfig


class LogDir:
    def __init__(self, path):
        self.path = path

    @property
    def checkpoints(self):
        ckptdir = os.path.join(self.path, 'ckpt')
        if os.path.exists(ckptdir):
            return sorted([int(p.split('.')[0]) for p in os.listdir(ckptdir)])
        return []

    @property
    def config(self):
        return AlphaZeroConfig(os.path.join(self.path, 'config.yaml'))

    def load_checkpoint(self, i):
        ckptdir = os.path.join(self.path, 'ckpt')
        if not os.path.exists(ckptdir):
            raise FileNotFoundError
        ckpts = {int(p.split('.')[0]): p for p in os.listdir(ckptdir)}
        if i not in ckpts:
            raise FileNotFoundError
        ckpt = torch.load(os.path.join(self.path, 'ckpt', ckpts[i]))
        model = self.config.Model()
        model.load_state_dict(ckpt['model_state_dict'])
        return model
