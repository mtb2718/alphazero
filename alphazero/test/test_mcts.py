import os

import numpy as np
import torch

from alphazero.config import AlphaZeroConfig
from alphazero.game import Game
from alphazero.mcts import MCTreeNode, run_mcts
from alphazero.replay_buffer import ReplayBufferDataset

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, 'configs/tictactoe.yaml')

def test_mcts_and_replay_buffer(tmp_path):

    NUM_SIMS_PER_ACTION = 64
    TEST_GAME_HISTORY = [0, 2, 8, 4, 6, 7, 3]

    # Simulate game-play, forcing moves from both players to ultimately result in
    # a terminal game configuration where plaeyr 0 has forced a win:
    #  X |   | O
    # -----------
    #  X | O |  
    # -----------
    #  X | O | X

    # Note that we're using a UniformModel, so the value
    # information should end up getting backpropagated.
    config = AlphaZeroConfig(CONFIG)
    model = config.Model()
    game = config.Game()
    dataset = ReplayBufferDataset(config, os.path.join(str(tmp_path), 'test.sqlite'))

    # Populate with search stats as we step through game
    # Should expect the values ultimately coming out of the replay buffer to match.
    search_stats = np.zeros((len(TEST_GAME_HISTORY), game.NUM_ACTIONS))

    for i, action in enumerate(TEST_GAME_HISTORY):
        root = MCTreeNode()
        run_mcts(game, root, model, NUM_SIMS_PER_ACTION)
        N = np.sum(root.num_visits)
        assert N == NUM_SIMS_PER_ACTION, \
            f'MCTS should run the specified number of simulations: {N} != {NUM_SIMS_PER_ACTION}'
        search_stats[i, game.valid_actions] = root.num_visits
        game.take_action(action, search_statistics=root.num_visits)

    # Insert game replay buffer
    dataset.add_game(game, 0)
    assert len(dataset) == len(TEST_GAME_HISTORY), \
        'Replay buffer should have an example for each non-terminal game state'

    # Test retrieval from replay buffer returns values we expect
    for i in range(len(dataset)):
        example = dataset[i]
        assert np.sum(example['p_valid']) == 9 - i, 'Expect history to be inserted sequentially'

        print(f"After {i} moves:")
        print(f"  Outcome: {example['z']}")
        print(f"  Psearch: {example['p']}")

        p_search = search_stats[i] / np.sum(search_stats[i])
        assert np.all(example['p'] == p_search)
        if i % 2 == 0:
            assert example['z'] == 1
        else:
            assert example['z'] == -1


TEST_GAME_SPEC = {
    (): {
        'inferred_value': 0.9,
        'inferred_prior': [1.],
    },
    (0,): { # P0 move 1
        'inferred_value': -0.9,
        'inferred_prior': [1.],
    },
    (0, 0): { # P1 move 1
        'inferred_value': 0.9,
        'inferred_prior': [1.],
    },
    (0, 0, 0): { # P0 move 2
        'inferred_value': -0.9,
        'inferred_prior': [1.],
    },
    (0, 0, 0, 0): { # P1 move 2
        'inferred_value': 0.9,
        'inferred_prior': [1.],
    },
    (0, 0, 0, 0, 0): { # P0 move 3, P0 wins
        'winner': 0
    },
}

class SimpleModel:
    def __init__(self, spec):
        self._spec = spec
    def eval(self):
        pass
    def __call__(self, x, p_valid):
        history = tuple(x.int().tolist()[0])
        assert history in self._spec
        p = self._spec[history]['inferred_prior']
        v = self._spec[history]['inferred_value']
        return torch.tensor([p]), torch.tensor([v])

class SimpleGame(Game):
    def __init__(self, spec):
        self.NUM_ACTIONS = len(spec[()]['inferred_prior'])
        super(SimpleGame, self).__init__()
        self._spec = spec
    def __str__(self):
        return ''
    @property
    def terminal(self):
        return self.winner is not None
    @property
    def winner(self):
        return self._spec[tuple(self.history)].get('winner')
    @property
    def valid_actions(self):
        return list(range(self.NUM_ACTIONS))
    def render(self, turn=None):
        return np.array(self.history, dtype=np.float32)

def print_tree(node, history=[]):
    if node.parent is None:
        print('Printing Tree from root:')
    print(f'History: {history}')
    if node.expanded:
        print('P:', node.action_prior)
        print('N:', node.num_visits)
        print('Q:', node.mean_value)
    else:
        return
    for i, n in enumerate(node.children):
        if n is not None:
            print_tree(n, history + [i])

def test_mcts_backprop():
    # Ensure that value assignment is consistent and correct during mcts backpropagation
    model = SimpleModel(TEST_GAME_SPEC)
    game = SimpleGame(TEST_GAME_SPEC)
    root = MCTreeNode()
    for i in range(20):
        print(f"Expanding 1 state, iter {i}")
        print(f"---------------------------")
        run_mcts(game, root, model, 1)
        print_tree(root)

        root_value = root.mean_value[0]
        if i < 4:
            # First 4 moves should all backprop 0.9 value to root state
            assert abs(root_value - 0.9) < 1e-6
        else:
            # Later simulations resulting in a player 0 win should begin
            # increasing value estimate at root.
            assert root_value > prev_root_value
        prev_root_value = root_value

        if i > 0:
            child_value = root.children[0].mean_value[0]
            if i < 4:
                assert abs(child_value + 0.9) < 1e-6
            else:
                assert child_value < prev_child_value
            prev_child_value = child_value

    # Ensure that running MCTS from a non-root node of an existing search tree,
    # as is the case when re-using the search tree during game play, does not
    # corrupt search statistics higher up in the tree.
    print("Expanding 1 state from root's child")
    child = root.children[0]
    game.take_action(0)
    run_mcts(game, child, model, 1)
    print_tree(child)
    assert abs(root.mean_value[0] - prev_root_value) < 1e-6, \
        'Root node should not be corrupted by search on child'
    assert child.mean_value[0] < prev_child_value, \
        'Child node search should improve statistics'
