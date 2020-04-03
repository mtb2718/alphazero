import numpy as np

from alphazero.games.tictactoe import TicTacToe
from alphazero.games.connectfour import ConnectFour


def test_connect_four():

    # History of a game with perfect play from both sides
    # From: https://connect4.gamesolver.org/?pos=44444666641267222263721335117357537731155
    # Note actions are 1-indexed
    PERFECT_GAME = '44444666641267222263721335117357537731155'
    actions = [int(c) - 1 for c in PERFECT_GAME]

    # Test outputs are all as expected for a long game
    game = ConnectFour()
    for i, a in enumerate(actions):
        print(game)
        assert a in game.valid_actions
        assert not game.terminal
        assert game.winner is None
        board = game.render()
        assert np.sum(board) == i
        assert np.sum(board[1]) - np.sum(board[0]) == i % 2
        s = str(game)
        assert s.count('X') == (i + 1) // 2
        assert s.count('O') == i // 2
        game.take_action(a)
    print(game)
    assert game.terminal
    assert game.winner == 0

    # Test 'winner' returns correct answer for a handful of positions
    TEST_GAME_POSITIONS = [
        # Vertical wins
        {'winner': 0, 'history': [0,1,0,1,0,1,0]},
        {'winner': 1, 'history': [3,0,1,0,1,0,1,0]},
        {'winner': 0, 'history': [6,1,6,1,6,1,6]},
        {'winner': 1, 'history': [3,6,1,6,1,6,1,6]},
        {'winner': 0, 'history': [3,3,3,2,3,2,3,2,3]},
        # Horizontal wins
        {'winner': 0, 'history': [0,0,1,1,2,2,3]},
        {'winner': 1, 'history': [4,0,0,1,1,2,2,3]},
        {'winner': 0, 'history': [6,6,5,5,4,4,3]},
        {'winner': 1, 'history': [0,6,6,5,5,4,4,3]},
        # Connect more-than-4
        {'winner': 0, 'history': [6,6,5,5,4,4,0,0,1,1,2,2,3]},
        {'winner': 1, 'history': [6,6,6,5,5,4,4,0,0,1,1,2,2,3]},
        # Diagonal wins, ascending
        {'winner': 0, 'history': [0,1,1,3,2,2,2,2,3,3,3]},
        {'winner': 1, 'history': [4,0,1,1,3,2,2,2,2,3,3,3]},
        {'winner': 0, 'history': [3,4,4,6,5,5,5,5,6,6,6]},
        {'winner': 1, 'history': [1,3,4,4,6,5,5,5,5,6,6,6]},
        # Diagonal wins, descending
        {'winner': 0, 'history': [1,0,0,0,0,2,2,1,1,5,3]},
        {'winner': 1, 'history': [0,0,0,0,1,1,2,1,5,2,5,3]},
    ]

    for pos in TEST_GAME_POSITIONS:
        game = ConnectFour(pos['history'][:-1])
        assert not game.terminal
        assert game.winner is None
        game.take_action(pos['history'][-1])
        print(game)
        assert game.terminal
        assert game.winner == pos['winner']


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
