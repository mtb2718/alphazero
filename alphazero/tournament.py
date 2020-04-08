from argparse import ArgumentParser
from itertools import product
import multiprocessing as mp
import os
import pickle

import numpy as np
from tqdm import tqdm
import torch

from alphazero.config import AlphaZeroConfig
from alphazero.eval import AlphaZeroPlayer, SolverPlayer, play


class LogDir:
    def __init__(self, path):
        self.path = path

    @property
    def checkpoints(self):
        ckptdir = os.path.join(self.path, 'ckpt')
        if os.path.exists(ckptdir):
            return [int(p.split('.')[0]) for p in os.listdir(ckptdir)]
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


def run_matchup(matchup):
    c0, c1 = matchup

    l0, i0 = c0
    m0 = l0.load_checkpoint(i0)
    m0 = m0.to(torch.device('cuda:0'))

    l1, i1 = c1
    m1 = l1.load_checkpoint(i1)
    m1 = m1.to(torch.device('cuda:1'))

    assert type(l0.config.Game()) == type(l1.config.Game()), 'Players must agree on game.'
    p0 = AlphaZeroPlayer(m0)
    p1 = AlphaZeroPlayer(m1)

    return c0, c1, play_series(l0.config.Game, p0, p1, 0)


def play_series(Game, player0, player1, N):
    outcome = {}

    # play one game greedily/deterministically
    game = Game()
    play(game, [player0, player1])
    outcome['greedy'] = {
        'history': game.history,
        'regret': regret(Game, game.history),
        'winner': game.winner,
    }

    # play N games with sampling
    outcome['sample'] = []
    player0.exploration = 1
    player1.exploration = 1
    for _ in range(N):
        game = Game()
        play(game, [player0, player1])
        outcome['sample'].append({
            'history': game.history,
            'regret': regret(Game, game.history),
            'winner': game.winner,
        })

    return outcome


def run_tournament():
    resultdir = '/data/alphazero/evals/connect4/20200331-perf-sync-1gpu'
    logdir = LogDir('/data/alphazero/connect4/20200331-perf-sync-1gpu')
    candidates = [(logdir, 1000 * i) for i in range(1, 11)]
    matchups = list(product(candidates, repeat=2))

    results = []
    with mp.Pool(5) as pool:
        for c0, c1, outcome in tqdm(pool.imap_unordered(run_matchup, matchups), total=len(matchups)):
            results.append({
                '0': (c0[0].path, c0[1]),
                '1': (c1[0].path, c1[1]),
                'outcome': outcome,
            })

    os.makedirs(resultdir, exist_ok=True)
    with open(os.path.join(resultdir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    return results


def run_solver_matchup(matchup):
    from alphazero.games.connectfour import ConnectFour
    c0, c1 = matchup
    p0 = SolverPlayer(temperature=c0)
    p1 = SolverPlayer(temperature=c1)

    N_GAMES = 100
    return c0, c1, play_series(ConnectFour, p0, p1, N_GAMES)


def run_solver_calibration():
    resultdir = '/data/alphazero/evals/connect4/solvers'

    candidates = [0.25, 0.5, 1, 2, 4]
    matchups = list(product(candidates, repeat=2))
    results = []
    with mp.Pool(5) as pool:
        for c0, c1, outcome in tqdm(pool.imap_unordered(run_solver_matchup, matchups), total=len(matchups)):
            results.append({
                '0': c0,
                '1': c1,
                'outcome': outcome,
            })

    os.makedirs(resultdir, exist_ok=True)
    with open(os.path.join(resultdir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    return results


def regret(Game, history):
    r = [0] * len(history)
    for i, action_taken in enumerate(history):
        g = Game(history=history[:i])
        scores, _ = g.solve()
        optimal_score = np.max(scores)
        action_index = g.valid_actions.index(action_taken)
        r[i] = optimal_score - scores[action_index]
    return r


if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args()

    run_solver_calibration()
    #results = run_tournament()

