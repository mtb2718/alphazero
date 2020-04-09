from argparse import ArgumentParser
from itertools import product
import os
import pickle

import torch

from alphazero.config import AlphaZeroConfig
from alphazero.eval import AlphaZeroPlayer, play


def load_ckpt(ckpt):
    config_path = os.path.join(os.path.dirname(ckpt), '../config.yaml')
    config = AlphaZeroConfig(config_path)
    model = config.Model()
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, config


def play_series(Game, player0, player1, N):
    result = {}

    # play one game greedily/deterministically
    game = Game()
    play(game, [player0, player1])
    result['greedy'] = game.history

    # play N games with sampling
    result['sample'] = []
    player0.exploration = 1
    player1.exploration = 1
    for _ in range(N):
        game = Game()
        play(game, [player0, player1])
        result['sample'].append(game.history)

    return result


if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args()

    candidates = []

    logdir = '/data/alphazero/connect4/20200331-perf-sync-1gpu'
    ckpts = os.listdir(os.path.join(logdir, 'ckpt'))
    kwargs = {}
    for ckpt in sorted(ckpts)[::2]:
        candidate = (logdir, ckpt, kwargs)
        candidates.append(candidate)

    tournament_results = []
    for c0, c1 in product(candidates, repeat=2):
        ckpt0 = os.path.join(c0[0], 'ckpt', c0[1])
        m0, conf0 = load_ckpt(ckpt0)
        m0 = m0.cuda()

        ckpt1 = os.path.join(c1[0], 'ckpt', c1[1])
        m1, conf1 = load_ckpt(ckpt1)
        m1 = m1.cuda()

        assert type(conf0.Game()) == type(conf1.Game()), 'Players must agree on game.'
        p0 = AlphaZeroPlayer(m0, **c0[2])
        p1 = AlphaZeroPlayer(m1, **c1[2])

        tournament_results.append({
            '0': c0,
            '1': c1,
            'outcome': play_series(conf0.Game, p0, p1, 16)
        })

    with open('tournament_results.pkl', 'wb') as f:
        pickle.dump(tournament_results, f)
