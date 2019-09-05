from itertools import product

import torch
import numpy as np
from scipy.signal import convolve2d


GRID_WIDTH = 7
GRID_HEIGHT = 6


class AlphaC4(torch.nn.Module):

    def __init__(self):
        super(AlphaC4, self).__init__()
        self._backbone = torch.nn.Sequential(
            torch.nn.Conv2d(2, 16, (3, 3), padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(16, 32, (3, 3), padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(32, 32, (3, 3), padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(32, 32, (3, 3), padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(32, 32, (3, 3), padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(32, 8, (1, 1), padding=0, stride=1, bias=False),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(True),
        )

        self._policy_head = torch.nn.Sequential(
            torch.nn.Conv2d(8, 6 * 7, (6, 7), padding=0, stride=1, bias=True),
            torch.nn.ReLU(True),
        )

        self._value_head = torch.nn.Sequential(
            torch.nn.Conv2d(8, 1, (6, 7), padding=0, stride=1, bias=True),
        )

    def forward(self, board):
        feat = self._backbone(board)
        p = self._policy_head(feat).reshape(-1, 1, 6, 7)
        v = self._value_head(feat)
        return p, v


class ConnectFourState:
    def __init__(self):
        self._history = []


    @property
    def turn(self):
        return len(self._history) % 2


    @property
    def board(self):
        # Render network input
        board = np.zeros((2, GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)
        num_filled = [0] * GRID_WIDTH
        for i, col in enumerate(self._history):
            player = i % 2
            board[player, num_filled[col], col] = 1
            num_filled[col] += 1
        return board


    @property
    def valid_actions(self):
        """Return list of valid actions player can take in this state.

        A valid action in this case is represented by column index.
        """
        return [i for i in range(GRID_WIDTH) if self._history.count(i) < GRID_HEIGHT]


    def eval(self, net):

        # Channel 0 always equals current player's position
        b = self.board
        if self.turn == 1:
            b[[0, 1]] = b[[1, 0]]

        # Run inference
        x = torch.from_numpy(b[np.newaxis, ...])
        p, v = net(x.float())
        p = p.numpy()
        v = float(v.numpy())

        # Post-processing:
        # Mask invalid outputs, re-normalize, map to 0-6 int action space
        valid_mask = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)
        for i in range(GRID_WIDTH):
            next_row = self._history.count(i)
            if next_row < GRID_HEIGHT:
                valid_mask[next_row, i] = 1

        p *= valid_mask
        if np.sum(p) > 1e-5:
            p /= np.sum(p)
        else:
            p = valid_mask / np.sum(valid_mask)
        p = np.sum(p, axis=(0, 1, 2))
        p = p[self.valid_actions]
        return p, v


    def copy(self):
        o = ConnectFourState()
        o._history = self._history.copy()
        return o


    def take_action(self, col):
        """Take current player's move.
        Args:
            col (int) - drop a piece in this column.
        Returns True on success, False for invalid moves.
        """
        if col in self.valid_actions:
            self._history.append(col)
            return True
        return False


    @property
    def winner(self):
        """Test whether there is a winner in this game state.

        Returns:
            winner (int or None) - returns None if the game is not over, otherwise
            returns the index of the winning player or -1 for a draw.
        """
        winner = None

        h0 = np.ones((1, 4)) / 4
        h1 = np.ones((4, 1)) / 4
        h2 = np.eye(4) / 4
        h3 = np.eye(4)[::-1] / 4

        board = self.board
        for p, h in product([0, 1], [h0, h1, h2, h3]):
            if np.any(convolve2d(board[p], h, 'valid') >= 1):
                assert winner is None, 'Game cannot have multiple winners.'
                winner = p

        if winner is None and len(self.valid_actions) == 0:
            return -1
        return winner


    def __str__(self):
        s = ''
        board = self.board
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


if __name__ == '__main__':
    s = ConnectFourState()
    states = [s]
    print(s)
    while s.winner is None:
        a = input(f'(Player {s.turn}) Take Action [0-{GRID_WIDTH - 1}, (u)ndo, (q)uit], (d)ebug: ')
        if a == 'q':
            exit(0)
        elif a == 'u':
            if len(states) > 1:
                s = states[-2]
                states = states[:-1]
        elif a == 'd':
            import pdb
            pdb.set_trace()
            continue
        elif int(a) in s.valid_actions:
            s = s.copy()
            if s.move(int(a)):
                states.append(s)
            else:
                print('Invalid move.')
                continue
        else:
            print('Invalid Selection.')
            continue
        print(s)
        if s.winner is None:
            continue
        elif s.winner >= 0:
            print(f'Player {s.winner} wins!')
        else:
            print('Tie!')
        exit(0)
