from itertools import product
import numpy as np
from scipy.signal import convolve2d


GRID_WIDTH = 7
GRID_HEIGHT = 6


class ConnectFourState:
    def __init__(self):
        self._board = np.zeros((2, GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)
        self._turn = 0

    @property
    def board(self):
        return self._board

    @property
    def turn(self):
        return self._turn

    @property
    def valid_moves(self):
        mask = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)
        cols = np.arange(GRID_WIDTH)
        h = np.sum(self._board, axis=(0,1))
        v = h < GRID_HEIGHT - 1
        mask[h[v], cols[v]] = 1
        return mask

    def copy(self):
        o = ConnectFourState()
        o._board = self._board.copy()
        o._turn = self._turn
        return o

    def move(self, col):
        """Take current player's move.
        Args:
            col (int) - drop a piece in this column.
        Returns True on success, False for invalid moves.
        """
        c = np.sum(self._board[:, :, col], axis=0)
        occupied = np.nonzero(c)[0]
        n = -1 if len(occupied) == 0 else occupied[-1]
        if n < GRID_HEIGHT - 1:
            self._board[self._turn, n + 1, col] = 1
            self._turn = (self._turn + 1) % 2
            return True
        return False

    def terminal(self):
        """Test whether this is a terminal game state.

        Returns:
            done (bool) - True if game is over
            draw (bool) - True if the game is a draw
            winner (int) - 0/1, indicating winner if done=True, else -1
        """
        done = False
        draw = False
        winner = -1

        h0 = np.ones((1, 4)) / 4
        h1 = np.ones((4, 1)) / 4
        h2 = np.eye(4) / 4
        h3 = np.eye(4)[::-1] / 4

        for p, h in product([0, 1], [h0, h1, h2, h3]):
            if np.any(convolve2d(self._board[p], h, 'valid') >= 1):
                done = True
                winner = p
                break

        if not done and np.all(np.sum(self._board, axis=0) > 0):
            done = True
            draw = True

        return done, draw, winner

    def __str__(self):
        s = ''
        for r in reversed(range(GRID_HEIGHT)):
            rowstr = '|'
            for c in range(GRID_WIDTH):
                if self._board[0, r, c] > 0:
                    rowstr += 'X'
                elif self._board[1, r, c] > 0:
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
    while True:
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
        elif a in [str(i) for i in range(GRID_WIDTH)]:
            s = s.copy()
            if s.move(int(a)):
                states.append(s)
            else:
                print('Invalid move.')
                continue
        else:
            print('Invalid Selection.')
            continue
        done, draw, winner = s.terminal()
        print(s)
        if done:
            if draw:
                print('Tie!')
            else:
                print(f'Player {winner} wins!')
            exit(0)
