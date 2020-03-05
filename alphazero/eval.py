from argparse import ArgumentParser
import importlib
import os

import numpy as np
import torch

from alphazero.config import AlphaZeroConfig
from alphazero.mcts import MCTreeNode, run_mcts


class Player: # TODO: ABC
    def __init__(self):
        pass
    def set_game(self, game):
        pass
    def get_action(self):
        pass
    def observe_action(self, action):
        pass


class HumanPlayer(Player):
    def __init__(self):
        super(HumanPlayer, self).__init__()
        self._game = None

    def set_game(self, game):
        self._game = game

    def get_action(self):
        print(self._game)
        a = input(f'(Player {self._game.next_player}) Take Action [{self._game.valid_actions}], (q)uit, (d)ebug: ')
        try:
            action = int(a)
            return action, None
        except:
            return None, a


class AlphaZeroPlayer(Player):
    # TODO: Merge this with the SelfPlayWorker class and add .train()/.eval() mode?
    def __init__(self, model, debug=False):
        super(AlphaZeroPlayer, self).__init__()
        self._model = model
        self._debug = debug
        self._game = None
        self._tree = None

    def set_game(self, game):
        self._game = game
        self._tree = MCTreeNode()
        run_mcts(game, self._tree, self._model, 0, epsilon=0)

    def get_action(self):
        if self._debug:
            print(self._game)
        run_mcts(self._game, self._tree, self._model, 16, epsilon=0)
        if self._debug:
            N = self._tree.num_visits
            P = self._tree.action_prior
            print(f'Num visits ({np.sum(N)}): {N}')
            print(f'Action Prior: {P}')
        action_index = self._tree.greedy_action()
        self._tree = self._tree.traverse(action_index)
        self._tree.kill_siblings()
        action = self._game.valid_actions[action_index]
        return action, None

    def observe_action(self, action):
        action_index = self._game.valid_actions.index(action)
        self._tree = self._tree.traverse(action_index)
        self._tree.kill_siblings()


def play(game, players, current_player_index=0):
    """Play game from given state with given players."""

    for player in players:
        player.set_game(game)

    while not game.terminal:

        player = players[current_player_index]
        action, game_control = player.get_action()

        if game_control is None:
            if action not in game.valid_actions:
                print('Invalid Action.')
                continue
            for p in players:
                if p != player:
                    p.observe_action(action)
            game.take_action(action)
            current_player_index = (current_player_index + 1) % len(players)
        elif game_control == 'q':
            exit(0)
        elif game_control == 'd':
            import pdb; pdb.set_trace()
            continue
        else:
            print('Invalid Selection.')
            continue

    print(game)
    if game.winner >= 0:
        print(f'Player {game.winner} wins!')
    else:
        print('Game drawn!')


def load_ckpt(ckpt):
    config_path = os.path.join(os.path.dirname(ckpt), 'config.yaml')
    config = AlphaZeroConfig(config_path)
    model = config.Model()
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, config


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-0', '--ckpt0',
        default=None,
        help="Path to a ckpt.pt file")
    parser.add_argument('-1', '--ckpt1',
        default=None,
        help="Path to a ckpt.pt file")
    parser.add_argument('-g', '--game',
        default=None,
        help="Import path of Game to play, if no checkpoints specified.")
    args = parser.parse_args()

    players = []
    configs = []

    Game = None
    if args.game:
        module_path = '.'.join(args.game.split('.')[:-1])
        module = importlib.import_module(module_path)
        Game = getattr(module, args.game.split('.')[-1])

    for i in (0, 1):
        ckpt = getattr(args, f'ckpt{i}', None)
        if ckpt:
            model, config = load_ckpt(ckpt)
            if not Game:
                Game = config.Game
            else:
                assert Game == config.Game, 'AlphaZero players must agree on game.'
            player = AlphaZeroPlayer(model)
        else:
            player = HumanPlayer()
            config = None
        players.append(player)
        configs.append(config)

    assert Game is not None, 'Game must be specified either in config or on the command line.'

    play(Game(), players)
