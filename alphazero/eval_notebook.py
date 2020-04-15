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
import numpy as np
import seaborn as sn

# Wide cells
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))

# Hot reload scripts
# %load_ext autoreload
# %autoreload 2

# +
pkl = '/data/alphazero/evals/connect4/20200331-perf-sync-1gpu/results.pkl'

with open(pkl, 'rb') as f:
    results = pickle.load(f)
results.sort(key=lambda r: (r['0'], r['1']))
# -



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



# +
import pickle

with open('matchups.pkl', 'wb') as f:
    pickle.dump(matchups, f)
# -


