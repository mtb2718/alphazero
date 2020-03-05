from itertools import product

import numpy as np
import torch

from alphazero.game import Game
from alphazero.models import AlphaZero, AlphaZeroLoss


class AlphaZeroTTT(AlphaZero):
    def __init__(self, num_blocks, channels_per_block):
        super(AlphaZeroTTT, self).__init__(shape_in=(2, 3, 3),
                                           shape_out=(9,),
                                           num_blocks=num_blocks,
                                           block_channels=channels_per_block)

    def forward(self, x, p_valid):
        p, v = super(AlphaZeroTTT, self).forward(x)
        return p, v


class AlphaZeroTTTLoss(AlphaZeroLoss):
    pass


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

    def __init__(self, history=None):
        super(TicTacToe, self).__init__(history)

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
