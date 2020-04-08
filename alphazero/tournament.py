from argparse import ArgumentParser
import importlib
from itertools import product
import os

import numpy as np
import torch

from alphazero.config import AlphaZeroConfig
from alphazero.eval import AlphaZeroPlayer, play_game
from alphazero.mcts import MCTreeNode, run_mcts

'''
class AlphaZeroPlayer(Player):
    # TODO: Merge this with the SelfPlayWorker class and add .train()/.eval() mode?
    def __init__(self, model, debug=False):
        super(AlphaZeroPlayer, self).__init__()
        self._model = model
        self._debug = debug
        self._game = None
        self._tree = None

    def set_game(self, game):
        self._game = game
        self._tree = MCTreeNode()
        run_mcts(game, self._tree, self._model, 0, epsilon=0)

    def get_action(self):
        if self._debug:
            print(self._game)
        run_mcts(self._game, self._tree, self._model, 128, epsilon=0)
        if self._debug:
            N = self._tree.num_visits
            P = self._tree.action_prior
            print(f'Num visits ({np.sum(N)}): {N}')
            print(f'Action Prior: {P}')
        action_index = self._tree.greedy_action()
        self._tree = self._tree.traverse(action_index)
        self._tree.kill_siblings()
        action = self._game.valid_actions[action_index]
        return action, None

    def observe_action(self, action):
        action_index = self._game.valid_actions.index(action)
        self._tree = self._tree.traverse(action_index)
        self._tree.kill_siblings()
'''


def load_ckpt(ckpt):
    config_path = os.path.join(os.path.dirname(ckpt), '../config.yaml')
    config = AlphaZeroConfig(config_path)
    model = config.Model()
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, config


def play_series(Game, player0, player1, N=64):
    # play one game greedily/deterministically
    play(Game(), [player0, player1])

    # play N games with sampling



if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args()

    candidates = []

    logdir = '/data/alphazero/connect4/20200331-perf-sync-1gpu'
    ckpts = os.listdir(os.path.join(logdir, 'ckpt'))
    kwargs = {}
    for ckpt in ckpts:
        candidate = (logdir, ckpt, kwargs)
        candidates.append(candidate)

    tournament_results = []
    for c0, c1 in product(ckpts, repeat=2):
        ckpt0 = os.path.join(c0[0], 'ckpt', c0[1])
        m0, conf0 = load_ckpt(ckpt0)
        m0 = m0.cuda()
        m0.eval()

        ckpt1 = os.path.join(c1[0], 'ckpt', c1[1])
        m1, conf1 = load_ckpt(ckpt1)
        m1 = m1.cuda()
        m1.eval()

        assert type(conf0.Game()) == type(conf1.Game()), 'Players must agree on game.'
        p0 = AlphaZeroPlayer(m0, **c0[2])
        p1 = AlphaZeroPlayer(m1, **c1[2])

        results = play_series(conf0.Game, p0, p1)

