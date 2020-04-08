from collections import namedtuple
import csv
import os
import subprocess as sp
from tempfile import NamedTemporaryFile


# Define pgn-like results format
# p0:p1:result:csv-history:csv-regret
GameOutcome = namedtuple('GameOutcome', ('player0', 'player1', 'result', 'history', 'regret'))


def load_outcomes(outcomes_file):
    def to_int_list(s):
        return [int(i) for i in s.split(',')]
    games = []
    with open(outcomes_file, 'r') as f:
        for game in f.read().splitlines():
            toks = game.split(':')
            result = int(toks[2])
            history = to_int_list(toks[3])
            regret = to_int_list(toks[4])
            games.append(GameOutcome(*toks[:2], result, history, regret))
    return games


def save_outcomes(outcomes, outcomes_file):
    with open(outcomes_file, 'w') as f:
        for outcome in outcomes:
            history = ','.join([str(i) for i in outcome.history])
            regret = ','.join([str(r) for r in outcome.regret])
            f.write(f'{outcome.player0}:{outcome.player1}:{outcome.result}:{history}:{regret}\n')


def pgnstr(outcomes):
    PGN_LINE_TMPL = '[White "{0}"][Black "{1}"][Result "{2}"] 1. c4 Nf6'
    s = ''
    for outcome in outcomes:
        r = {0: "1-0", 1: "0-1", -1: "1/2-1/2"}[outcome.result]
        s += PGN_LINE_TMPL.format(outcome.player0, outcome.player1, r) + '\n'
    return s


def bayeselo(outcomes):
    """
        results - list of (player 0, player 1, winner) tuples
    """

    # Make temporary .pgn file for program input
    pgnfile = NamedTemporaryFile('wt', delete=False)
    pgnfile.write(pgnstr(outcomes))
    pgnfile.close()

    # Run and retrieve results
    ratingsfile = NamedTemporaryFile('rt', delete=False)
    ratingsfile.close()
    CMDS = [
        'prompt off',
        f'readpgn {pgnfile.name}',
        'elo',
        'minelo 0',
        'mm 1 1',
        f'advantage >{ratingsfile.name}',
        f'drawelo >>{ratingsfile.name}',
        f'ratings >>{ratingsfile.name}',
    ]
    input_str = '\n'.join(CMDS) + '\n'
    proc = sp.run('bayeselo',
                  input=input_str,
                  text=True,
                  stderr=sp.DEVNULL,
                  stdout=sp.DEVNULL)
    proc.check_returncode()

    # Parse and return results
    with open(ratingsfile.name, 'rt') as f:
        output_lines = f.read().splitlines()
        advantage_elo = float(output_lines[0])
        draw_elo = float(output_lines[1])
        ratings = {}
        for rating in output_lines[3:]:
            toks = rating.split()
            ratings[toks[1]] = float(toks[2])
    os.unlink(pgnfile.name)
    os.unlink(ratingsfile.name)
    return advantage_elo, draw_elo, ratings
