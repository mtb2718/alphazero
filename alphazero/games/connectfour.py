from itertools import product
import subprocess as sp

import numpy as np
from scipy.signal import convolve2d
import torch

from alphazero.models import AlphaZero
from alphazero.game import Game


GRID_WIDTH = 7
GRID_HEIGHT = 6


class AlphaZeroC4(AlphaZero):
    def __init__(self, num_blocks, channels_per_block):
        super(AlphaZeroC4, self).__init__(shape_in=(2, GRID_HEIGHT, GRID_WIDTH),
                                          shape_out=(GRID_HEIGHT, GRID_WIDTH),
                                          num_blocks=num_blocks,
                                          block_channels=channels_per_block)

    def forward(self, x, p_valid):
        p, v = super(AlphaZeroC4, self).forward(x, p_valid)
        filled_cells = torch.sum(x, dim=1)
        assert torch.all((filled_cells == 0) | (filled_cells == 1))
        next_row_in_col = torch.sum(filled_cells, dim=1).long()
        valid_actions_mask = next_row_in_col < GRID_HEIGHT
        assert torch.all(valid_actions_mask == p_valid)

        # Mask invalid grid cells, squash to BxA
        # TODO: Should be a fancy vectorized way to do this
        B = p.shape[0]
        valid_cells = torch.zeros_like(p)
        for b, c in product(range(B), range(GRID_WIDTH)):
            r = next_row_in_col[b, c]
            if r < GRID_HEIGHT:
                valid_cells[b, r, c] = 1
        p *= valid_cells
        p = torch.sum(p, dim=1)
        return p, v


class ConnectFour(Game):

    NUM_ACTIONS = GRID_WIDTH

    @property
    def valid_actions(self):
        return [i for i in range(GRID_WIDTH) if self.history.count(i) < GRID_HEIGHT]

    @property
    def terminal(self):
        if len(self.history) == GRID_WIDTH * GRID_HEIGHT:
            return True
        elif len(self.history) < 7:
            return False
        return self.winner is not None

    @property
    def winner(self):
        """Test whether there is a winner in this game state.

        Returns:
            winner (int or None) - returns None if the game is not over, otherwise
            the index of the winning player or -1 for a draw.
        """

        h0 = np.ones((1, 4))
        h1 = np.ones((4, 1))
        h2 = np.eye(4)
        h3 = np.eye(4)[::-1]

        board = self.render()
        win_channel = None
        winner = None
        for c, h in product([0, 1], [h0, h1, h2, h3]):
            if np.any(convolve2d(board[c], h, 'valid') >= 4):
                assert winner is None or winner == p, 'Game cannot have multiple winners.'
                win_channel = c
        if win_channel is None and len(self.valid_actions) == 0:
            return -1
        elif win_channel == 1:
            return self.prev_player
        elif win_channel == 0:
            assert False, "Previous player's move can never result in loss"
        return None

    def render(self):
        board = np.zeros((2, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
        num_filled = [0] * GRID_WIDTH
        for i, col in enumerate(self.history):
            action_player = i % 2
            j = 0 if action_player == self.next_player else 1
            board[j, num_filled[col], col] = 1
            num_filled[col] += 1
        return board

    def __str__(self):
        s = ''
        board = self.render()
        for r in reversed(range(GRID_HEIGHT)):
            rowstr = '|'
            for c in range(GRID_WIDTH):
                if board[0, r, c] > 0:
                    rowstr += 'X'
                elif board[1, r, c] > 0:
                    rowstr += 'O'
                else:
                    rowstr += ' '
                rowstr += '|'
            s += rowstr + '\n'
        s += ' ' + ' '.join([str(i) for i in range(GRID_WIDTH)])
        return s

    def solve(self):
        SOLVER = '/usr/local/bin/c4solver'
        BOOK = '/usr/local/bin/7x6.book'
        assert not self.terminal

        # Calculate value for all possible next states
        s0 = ''.join([str(a + 1) for a in self.history])
        input_str = s0 + '\n'
        for a in self.valid_actions:
            input_str += s0 + str(a + 1) + '\n' # Actions are 1-indexed in solver
        proc = sp.run([SOLVER, '-b', BOOK],
                      input=input_str,
                      text=True,
                      stdout=sp.PIPE,
                      stderr=sp.STDOUT)
        proc.check_returncode()
        stdout_lines = proc.stdout.splitlines()
        get_score = lambda line: int(line.split(' ')[1])
        scores = []
        my_score = get_score(stdout_lines[1])
        for line in stdout_lines[2:]:
            if len(line) == 0:
                continue
            if 'Invalid' in line:
                score = my_score
            else:
                score = -get_score(line)
            scores.append(score)
        v = 1 if my_score > 0 else -1 if my_score < 0 else 0
        return scores, v
