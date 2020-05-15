from argparse import ArgumentParser
from itertools import product
import multiprocessing as mp
import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sn
import torch
from tqdm import tqdm

from alphazero.agents import AlphaZeroPlayer, SolverPlayer
from alphazero.config import AlphaZeroConfig
from alphazero.elo import bayeselo, GameOutcome, load_outcomes, save_outcomes
from alphazero.eval import play
from alphazero.logdir import LogDir
from alphazero.tournament import regret


def run_matchup(job):
    (logdir, ckpt) = job[0]
    solver_temp = job[1]

    solver_player = SolverPlayer(temperature=solver_temp)
    model = logdir.load_checkpoint(ckpt)
    wid = int(mp.current_process().name.split('-')[1])
    ngpu = torch.cuda.device_count()
    if ngpu > 0:
        # TODO: Debug model dups on both GPUs / ckpt loading
        model = model.to(torch.device(f'cuda:{wid % ngpu}'))
    a0player = AlphaZeroPlayer(model)
    # TODO: Put logidr in path name
    pname = f'ckpt_{ckpt}'

    Game = logdir.config.Game
    outcomes = []

    # solver black
    game = Game()
    play(game, [a0player, solver_player])
    r = regret(Game, game.history)
    outcome = GameOutcome(pname, solver_temp, game.winner, game.history, r)
    outcomes.append(outcome)

    # solver white
    game = Game()
    play(game, [solver_player, a0player])
    r = regret(Game, game.history)
    outcome = GameOutcome(solver_temp, pname, game.winner, game.history, r)
    outcomes.append(outcome)

    return outcomes


def run_matchups(logdir, ckpts, solver_temps, num_games):
    candidates = [(logdir, ckpt) for ckpt in ckpts]
    matchups = list(product(candidates, solver_temps))

    jobs = matchups * num_games
    outcomes = []
    with mp.Pool() as pool:
        for matchup_outcomes in tqdm(pool.imap_unordered(run_matchup, jobs), total=len(jobs)):
            outcomes.extend(matchup_outcomes)
    return outcomes


def plot_ratings(outdir, solver_outcomes, eval_outcomes):

    # Calculate ratings
    advantage_elo, draw_elo, ratings = bayeselo(solver_outcomes + eval_outcomes)

    players = []
    player_ratings = []
    solvers = []
    solver_ratings = []
    for p, r in ratings.items():
        if p.startswith('ckpt'):
            players.append(int(p[5:]))
            player_ratings.append(r)
        else:
            solvers.append(p)
            solver_ratings.append(r)
    players, player_ratings = list(zip(*sorted(zip(players, player_ratings))))
    players = list(players)
    solvers, solver_ratings = list(zip(*reversed(sorted(zip(solvers, solver_ratings)))))
    solvers = list(solvers)

    # Create ELO vs training iter plot
    fig = plt.figure(figsize=(24,8))
    plt.plot(players, player_ratings)
    ax = fig.axes[0]
    plt.grid()
    plt.xticks([0] + players[3::4], rotation=90)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    xlim = [0, players[-1] + 1]
    plt.xlim(xlim)
    plt.title('AlphaZero vs Solvers')
    plt.xlabel('Training Step')
    plt.ylabel('BayesELO Rating')
    for s, r in zip(solvers, solver_ratings):
        plt.plot(xlim, [r, r], 'r--')
        plt.text(xlim[0], r, s, color='r')
    plt.savefig(f'{outdir}/ELO_ratings.png')

    # Create tabular heatmaps of wins/draws/losses
    Np = len(players)
    Ns = len(solvers)
    wij = np.zeros((Np, Ns), dtype=np.int32)
    lij = np.zeros((Np, Ns), dtype=np.int32)
    dij = np.zeros((Np, Ns), dtype=np.int32)
    for outcome in eval_outcomes:
        if outcome.player0.startswith('ckpt'):
            player_pos = 0
            i = players.index(int(outcome.player0[5:]))
            j = solvers.index(outcome.player1)
        elif outcome.player1.startswith('ckpt'):
            player_pos = 1
            i = players.index(int(outcome.player1[5:]))
            j = solvers.index(outcome.player0)
        else:
            assert False
        if outcome.result == player_pos:
            wij[i, j] += 1
        elif outcome.result == -1:
            dij[i, j] += 1
        else:
            lij[i, j] += 1
    vmax = np.max(wij + lij + dij)
    Sx = Ns * 3 + 3
    Sy = Np / 4
    fig = plt.figure(figsize=(Sx, Sy))
    plt.subplot(1, 3, 1)
    sn.heatmap(wij, 0, vmax, center=0, xticklabels=solvers, yticklabels=players, fmt='d', annot=True)
    plt.title('Win count')
    plt.subplot(1, 3, 2)
    sn.heatmap(dij, 0, vmax, center=0, xticklabels=solvers, yticklabels=players, fmt='d', annot=True)
    plt.title('Draw count')
    plt.subplot(1, 3, 3)
    sn.heatmap(lij, 0, vmax, center=0, xticklabels=solvers, yticklabels=players, fmt='d', annot=True)
    plt.title('Loss count')
    plt.savefig(f'{outdir}/outcomes.png')

def run_evaluation(logdir, ckpts, solver_outcomes, outdir, num_games_per_matchup):
    solver_temps = sorted(set([float(g.player0) for g in solver_outcomes]))

    #eval_outcomes = run_matchups(logdir, ckpts, solver_temps, num_games_per_matchup)
    #save_outcomes(eval_outcomes, f'{outdir}/eval_outcomes.txt')
    eval_outcomes = load_outcomes(f'{outdir}/eval_outcomes.txt')

    # dump results in pickle
    plot_ratings(outdir, solver_outcomes, eval_outcomes)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--ckpt',
        nargs=2,
        help='Checkpoint to evaluate.')
    parser.add_argument('-d', '--ckptdir',
        help='Checkpoint directory to evaluate.')
    parser.add_argument('-N', '--num-games-per-matchup',
        type=int,
        default=25,
        help='Number of games to play in each matchup.')
    parser.add_argument('-o', '--outdir',
        required=True,
        help='Directory to write evaluation results.')
    args = parser.parse_args()

    if args.ckpt is None and args.ckptdir is None:
        print('One of \'ckpt\' or \'ckptdir\' must be specified.')
        parser.print_usage()
        exit(1)

    if args.ckptdir:
        logdir = LogDir(args.ckptdir)
        ckpts = logdir.checkpoints
    else:
        logdir = LogDir(args.ckpt[0])
        ckpts = [int(args.ckpt[1])]

    # TODO: Load pre-existing results if given

    # TODO: ELO plot of multiple results

    # TODO: Load solver baselines from CLI
    SOLVER_RESULTS_TXT = '/data/alphazero/evals/connect4/solvers/results.txt'
    solver_outcomes = load_outcomes(SOLVER_RESULTS_TXT)

    os.makedirs(args.outdir, exist_ok=True)
    run_evaluation(logdir, ckpts, solver_outcomes, args.outdir, args.num_games_per_matchup)
