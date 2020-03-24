
from alphazero.games.tictactoe import TicTacToe
from alphazero.games.connectfour import ConnectFour


def test_tic_tac_toe_solver():
    HISTORY = [0, 2, 8, 4, 6, 7, 3]
    MOVE_SCORE_HISTORY = [
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0], # Empty board
        [    -2, -2, -2,  0, -2, -2, -2, -2], # 0
        [    -1,      2,  0,  0,  2,  0,  2], # 0, 2
        [    -3,     -3, -2, -3, -3, -3,   ], # 0, 2, 8
        [    -2,     -2,     -2,  2, -2,   ], # 0, 2, 8, 4
        [    -2,     -2,     -2,     -2,   ], # 0, 2, 8, 4, 6
        [     0,      2,     -1,           ], # 0, 2, 8, 4, 6, 7
        [                                  ], # 0, 2, 8, 4, 6, 7, 3
    ]

    for i, a in enumerate(HISTORY):
        game = TicTacToe(HISTORY[:i])
        print('=============')
        print('Solving game:')
        print(game)
        move_scores = MOVE_SCORE_HISTORY[i]
        assert len(game.valid_actions) == len(move_scores)
        solver_scores, v = game.solve()
        print('True solution:', move_scores)
        print('Solv solution:', solver_scores)
        assert move_scores == solver_scores, f'Move scores should match ref after {i} turns'


def test_connect_four_solver():

    # Reference values calculated by:
    # https://connect4.gamesolver.org/?pos=44443544231
    HISTORY = [3, 3, 3, 3, 2, 4, 3, 3, 1, 2, 0]
    MOVE_SCORE_HISTORY = [
        [-2,  -1,   0,   1,   0,  -1,  -2], # Empty board
        [-4,  -2,  -2,  -1,  -2,  -2,  -4], # 3
        [-3,  -3,  -2,   1,  -2,  -3,  -3], # 3, 3
        [-4,  -4,  -3,  -1,  -3,  -4,  -4], # 3, 3, 3
        [-2,  -2,  -2,   1,  -2,  -2,  -2], # 3, 3, 3, 3
        [-17,  2, -17, -17,   2, -17, -17], # 3, 3, 3, 3, 2
        [-2,  -2,  -3,  -3,  -2,  -3,  -4], # 3, 3, 3, 3, 2, 4
        [-1,   2,   0,  -2,   3,  -2,  -3], # 3, 3, 3, 3, 2, 4, 3
        [-1,   1,  -2,        2,  -2,  -2], # 3, 3, 3, 3, 2, 4, 3, 3
        [-1, -16, -16,      -16, -16, -16], # 3, 3, 3, 3, 2, 4, 3, 3, 1
        [16,   2,   2,        2,  -2,   0], # 3, 3, 3, 3, 2, 4, 3, 3, 1, 2
        [                                ], # 3, 3, 3, 3, 2, 4, 3, 3, 1, 2, 0 (X win)
    ]

    for i, a in enumerate(HISTORY):
        game = ConnectFour(HISTORY[:i])
        print('=============')
        print('Solving game:')
        print(game)
        move_scores = MOVE_SCORE_HISTORY[i]
        print(move_scores)
        assert len(game.valid_actions) == len(move_scores)
        solver_scores, v = game.solve()
        assert move_scores == solver_scores, f'Move scores should match ref after {i} turns'
