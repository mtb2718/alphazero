from itertools import product
import subprocess as sp

import numpy as np
import torch

from alphazero.models import AlphaZero
from alphazero.game import Game

# Game rule constants
GRID_WIDTH = 7
GRID_HEIGHT = 6
STREAK_LEN = 4
assert STREAK_LEN == 4, 'Only ConnectFour supported.'
N_SPACES = GRID_HEIGHT * GRID_WIDTH

# Precompute winning positions for fast game state evaluation
N_HORIZONTAL_WINS = GRID_HEIGHT * (GRID_WIDTH - STREAK_LEN + 1)
N_VERTICAL_WINS = GRID_WIDTH * (GRID_HEIGHT - STREAK_LEN + 1)
N_DIAG_WINS = 2 * (GRID_WIDTH - STREAK_LEN + 1) * (GRID_HEIGHT - STREAK_LEN + 1)
NUM_WINNING_POSITIONS = N_HORIZONTAL_WINS + N_VERTICAL_WINS + N_DIAG_WINS

WINNING_POSITIONS = np.zeros((NUM_WINNING_POSITIONS, N_SPACES), dtype=np.uint8)
n = 0
for row, col in product(range(GRID_HEIGHT), range(GRID_WIDTH)):
    l_right = col <= GRID_WIDTH - STREAK_LEN
    l_above = row <= GRID_HEIGHT - STREAK_LEN
    # horizontal wins
    if l_right:
        for i in range(STREAK_LEN):
            WINNING_POSITIONS[n, row * GRID_WIDTH + col + i] = 1
        n += 1
    # vertical wins
    if l_above:
        for i in range(STREAK_LEN):
            WINNING_POSITIONS[n, (row + i) * GRID_WIDTH + col] = 1
        n += 1
    # diagonal wins
    if l_right and l_above:
        for i in range(STREAK_LEN):
            WINNING_POSITIONS[n + 0, (row + i) * GRID_WIDTH + col + i] = 1
            WINNING_POSITIONS[n + 1, (row + STREAK_LEN - i - 1) * GRID_WIDTH + col + i] = 1
        n += 2
assert np.all(np.sum(WINNING_POSITIONS, axis=1) == STREAK_LEN)


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
        if len(self.history) < 2 * STREAK_LEN - 1:
            return False
        if len(self.history) == N_SPACES:
            return True
        return self.winner is not None

    @property
    def winner(self):
        """Test whether there is a winner in this game state.

        Returns:
            winner (int or None) - returns None if the game is not over, otherwise
            the index of the winning player or -1 for a draw.
        """

        if len(self.history) < 2 * STREAK_LEN - 1:
            return None

        # At any given game state, only the previous player could have won
        board = self.render()[1].reshape(N_SPACES)
        streak_lens = WINNING_POSITIONS @ board
        if np.max(streak_lens) >= STREAK_LEN:
            return self.prev_player

        if len(self.history) == N_SPACES:
            return -1

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
        c0 = 'X' if self.next_player == 0 else 'O'
        c1 = 'X' if self.prev_player == 0 else 'O'
        for r in reversed(range(GRID_HEIGHT)):
            rowstr = '|'
            for c in range(GRID_WIDTH):
                if board[0, r, c] > 0:
                    rowstr += c0
                elif board[1, r, c] > 0:
                    rowstr += c1
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
