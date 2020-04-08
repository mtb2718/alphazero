# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sn

from alphazero.elo import GameOutcome, load_outcomes, pgnstr, bayeselo

# Wide cells
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))

# Hot reload scripts
# %load_ext autoreload
# %autoreload 2

# +
solver_outcomes = load_outcomes('/data/alphazero/evals/connect4/solvers/results.txt')

CKPT_DIRS = [
    '20200331-perf-sync-1gpu',
    '20200403-db0',
    '20200404-db1-b512',
]


ckpt_outcomes = {d: load_outcomes(f'/data/alphazero/connect4/{d}/eval/eval_outcomes.txt') for d in CKPT_DIRS}

def _fixname(d, name):
    if 'ckpt' in name:
        ckpt = name.split('_')[-1]
        return f'{d}_{ckpt}'
    else:
        return f'solver_{name}'
print(ckpt_outcomes['20200403-db0'][0])
for d, outcomes in ckpt_outcomes.items():
    for i, o in enumerate(outcomes):
        outcomes[i] = o._replace(player0=_fixname(d, o.player0), player1=_fixname(d, o.player1))
print(ckpt_outcomes['20200403-db0'][0])


# Need to change name
all_outcomes = [o._replace(player0=_fixname('', o.player0), player1=_fixname('', o.player1)) for o in solver_outcomes]
for ckpt_outcome in ckpt_outcomes.values():
    all_outcomes.extend(ckpt_outcome)

advantage_elo, draw_elo, ratings = bayeselo(all_outcomes)

print('Advantage ELO:', advantage_elo)
print('Draw ELO:', draw_elo)

players = {}
player_ratings = {}
solvers = []
solver_ratings = []
for p, r in ratings.items():
    toks = p.split('_')
    suffix = toks[-1]
    name = '_'.join(toks[:-1])
    if name == 'solver':
        solvers.append(suffix)
        solver_ratings.append(r)
    else:
        if name not in players:
            players[name] = []
            player_ratings[name] = []
        players[name].append(int(suffix))
        player_ratings[name].append(r)

# sort
max_ckpt = 0
for logdir in players.keys():
    players[logdir], player_ratings[logdir] = list(zip(*sorted(zip(players[logdir], player_ratings[logdir]))))
    players[logdir] = list(players[logdir])
    max_ckpt = max(max_ckpt, int(players[logdir][-1]))

solvers, solver_ratings = list(zip(*reversed(sorted(zip(solvers, solver_ratings)))))
solvers = list(solvers)

xlim = [0, max_ckpt + 1]
fig = plt.figure(figsize=(24,8))
for logdir in sorted(players.keys()):
    plt.plot(players[logdir], player_ratings[logdir])
ax = fig.axes[0]
plt.grid()
plt.xticks([0] + players[list(players.keys())[0]][3::4], rotation=90)
ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
plt.xlim(xlim)
plt.title('AlphaZero vs Solvers')
plt.xlabel('Training Step')
plt.ylabel('BayesELO Rating')
plt.legend(sorted(players.keys()))

for s, r in zip(solvers, solver_ratings):
    plt.plot(xlim, [r, r], 'r--')
    plt.text(xlim[0], r, s, color='r')
fig.show()

# +
# from alphazero.elo import GameOutcome, load_outcomes, save_outcomes, pgnstr, bayeselo

# with open('/data/alphazero/evals/connect4/solvers/results.pkl', 'rb') as f:
#     results = pickle.load(f)
# results.sort(key=lambda r: (r['0'], r['1']))
# outcomes = []
# for matchup in results:
#     p0, p1 = matchup['0'], matchup['1']
#     for game in matchup['outcome']['sample']:
#         rst = int(game['winner'])
#         rgt = [-s for s in game['suboptimality']]
#         history = [i.item() for i in game['history']]
#         outcome = GameOutcome(float(p0), float(p1), rst, history, rgt)
#         outcomes.append(outcome)
# save_outcomes(outcomes, '/data/alphazero/evals/connect4/solvers/results.txt')


# +
Np = len(players)
Ns = len(solvers)

wij = np.zeros((Np, Ns), dtype=np.int32)
lij = np.zeros((Np, Ns), dtype=np.int32)
dij = np.zeros((Np, Ns), dtype=np.int32)

for outcome in outcomes:
    if outcome.player0.startswith('ckpt'):
        player_pos = 0
        i = players.index(int(outcome.player0[5:]))
        j = solvers.index(outcome.player1)
    elif outcome.player1.startswith('ckpt'):
        player_pos = 1
        i = players.index(int(outcome.player1[5:]))
        j = solvers.index(outcome.player0)
    else:
        continue

    if outcome.result == player_pos:
        wij[i, j] += 1
    elif outcome.result == -1:
        dij[i, j] += 1
    else:
        lij[i, j] += 1

vmax = 10
Sx = len(solvers) * 3 + 3
Sy = len(players) / 4

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
fig.show()




# +
ckpts = sorted(set([r['0'][1] for r in results]))
L = len(ckpts)

print(ckpts)

winner = np.zeros((L, L), dtype=np.int32)
winning_seat = np.zeros(L, dtype=np.int32)
so = np.zeros((L, L), dtype=np.float32)
p0so = np.zeros((L, L), dtype=np.float32)
p1so = np.zeros((L, L), dtype=np.float32)
gamelen = np.zeros((L, L), dtype=np.int32)

for n, r in enumerate(results):
    i, j = n // L, n % L
    outcome = r['outcome']['greedy']

    winner[i, j] = outcome['winner']
    if i == j:
        winning_seat[i] = outcome['winner']
    so[i, j] = sum(outcome['suboptimality']) / len(outcome['history'])
    p0so[i, j] = sum(outcome['suboptimality'][0::2]) / len(outcome['history'][0::2])
    p1so[i, j] = sum(outcome['suboptimality'][1::2]) / len(outcome['history'][1::2])
    gamelen[i, j] = len(outcome['history'])


plt.figure()
plt.plot(winning_seat)

plt.figure(figsize=(16, 16))
sn.heatmap(winner, center=0, xticklabels=ckpts, yticklabels=ckpts, fmt='d', annot=True, square=True)

plt.figure(figsize=(16, 16))
sn.heatmap(gamelen, center=0, xticklabels=ckpts, yticklabels=ckpts, fmt='d', annot=True, square=True)

plt.figure(figsize=(16, 16))
sn.heatmap(so, center=0, xticklabels=ckpts, yticklabels=ckpts, fmt='0.2f', annot=True, square=True)

plt.figure(figsize=(16, 16))
sn.heatmap(p0so, center=0, xticklabels=ckpts, yticklabels=ckpts, fmt='0.2f', annot=True, square=True)

plt.figure(figsize=(16, 16))
sn.heatmap(p1so, center=0, xticklabels=ckpts, yticklabels=ckpts, fmt='0.2f', annot=True, square=True)

plt.figure(figsize=(16, 16))
sn.heatmap(p0so - p1so, center=0, xticklabels=ckpts, yticklabels=ckpts, fmt='0.2f', annot=True, square=True)

