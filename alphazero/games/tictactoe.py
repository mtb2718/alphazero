from itertools import product

import numpy as np
import torch

from alphazero.game import Game


class TicTacToe(Game):
    """Squares are numbered 0-8 from top-left corner to bottom right corner."""

    NUM_ACTIONS = 9

    WINNING_POSITIONS = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],
        [0, 4, 8],
        [2, 4, 6]
    ]

    @property
    def terminal(self):
        if len(self.history) < 5:
            return False
        elif len(self.history) == 9:
            return True
        elif self.winner is not None:
            return True
        return False

    @property
    def winner(self):
        if len(self.history) < 5:
            return None
        for player, pos in product((0, 1), TicTacToe.WINNING_POSITIONS):
            if len(set(self.history[player::2]) & set(pos)) == 3:
                return player
        if len(self.history) == 9:
            return -1
        return None

    @property
    def valid_actions(self):
        return sorted(set(range(9)) - set(self.history))

    def render(self, turn=None):
        if turn is None:
            history = self.history
        else:
            assert turn <= len(self.history)
            assert turn >= 0
            history = self.history[:turn]
        img = np.zeros((2, 9), dtype=np.float32)
        next_player_actions = history[self.next_player::2]
        prev_player_actions = history[self.prev_player::2]
        img[0, next_player_actions] = 1
        img[1, prev_player_actions] = 1
        return img.reshape(2, 3, 3)

    def __str__(self):
        h = self.history
        tui = [' ' if i not in h else 'X' if h.index(i) % 2 == 0 else 'O' for i in range(9)]
        s  = ' ' + ' | '.join(tui[0:3]) + '\n'
        s += '-' * 11 + '\n'
        s += ' ' + ' | '.join(tui[3:6]) + '\n'
        s += '-' * 11 + '\n'
        s += ' ' + ' | '.join(tui[6:9])
        return s

    def solve(self):
        BOOK = {
            ():   (0, [ 0,  0,  0,  0,  0,  0,  0,  0,  0]),
            (0,): (0, [    -2, -2, -2,  0, -2, -2, -2, -2]),
            (1,): (0, [ 0,      0, -2,  0, -2, -2,  0, -2]),
            (2,): (0, [-2, -2,     -2,  0, -2, -2, -2, -2]),
            (3,): (0, [ 0, -2, -2,      0,  0,  0, -2, -2]),
            (4,): (0, [ 0, -2,  0, -2,     -2,  0, -2,  0]),
            (5,): (0, [-2, -2,  0,  0,  0,     -2, -2,  0]),
            (6,): (0, [-2, -2, -2, -2,  0, -2,     -2, -2]),
            (7,): (0, [-2,  0, -2, -2,  0, -2,  0,      0]),
            (8,): (0, [-2, -2, -2, -2,  0, -2, -2, -2    ]),
        }
        def minimax(game, root_depth=None, book={}):
            if root_depth is None:
                root_depth = len(game)

            if tuple(game.history) in book:
                return book[tuple(game.history)]

            root_player = root_depth % 2
            cur_player = len(game) % 2

            if game.terminal:
                nmoves = 5 - len(game) // 2
                outcome = game.terminal_value(root_player)
                return nmoves * outcome, []

            scores = []
            for a in game.valid_actions:
                sub_game = game.clone()
                sub_game.take_action(a)
                s, _ = minimax(sub_game, root_depth, book)
                scores.append(s)

            if cur_player == root_player:
                s = max(scores)
            else:
                s = min(scores)
            return s, scores

        s, scores = minimax(self, book=BOOK)
        v = 1 if s > 0 else -1 if s < 0 else 0
        return scores, v
