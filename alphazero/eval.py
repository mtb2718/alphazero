from argparse import ArgumentParser
import importlib
import os

import numpy as np
import torch

from alphazero.agents import AlphaZeroPlayer, HumanPlayer, SolverPlayer
from alphazero.config import AlphaZeroConfig

np.set_printoptions(suppress=True)


def play(game, players, show=False):
    """Play game from given state with given players."""

    for player in players:
        player.set_game(game)
        if type(player) == HumanPlayer:
            show = True

    while not game.terminal:
        player = players[game.next_player]
        if show:
            print('\nMoves taken:', len(game.history))
            print('---------------\n')
            print(game)
        action, game_control = player.get_action()

        if game_control is None:
            if action not in game.valid_actions:
                print('Invalid Action.')
                continue
            for p in players:
                if p != player:
                    p.observe_action(action)
            game.take_action(action)
        elif game_control == 'q':
            exit(0)
        elif game_control == 'd':
            import pdb; pdb.set_trace()
            continue
        else:
            print('Invalid Selection.')
            continue

    if show:
        if game.winner >= 0:
            print(f'\nPlayer {game.winner} wins!\n')
        else:
            print('\nGame drawn!\n')
        print(game)


def load_ckpt(ckpt):
    config_path = os.path.join(os.path.dirname(ckpt), '../config.yaml')
    config = AlphaZeroConfig(config_path)
    model = config.Model()
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, config


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-0', '--player0',
        nargs='+',
        default=[None],
        help="Path to a ckpt.pt file, or 'solver'")
    parser.add_argument('-1', '--player1',
        nargs='+',
        default=[None],
        help="Path to a ckpt.pt file, or 'solver")
    parser.add_argument('-g', '--game',
        default=None,
        help="Import path of Game to play, if no checkpoints specified.")
    parser.add_argument('--show',
        action='store_true',
        help='Show gameplay TUI. Default True if there are any human players.')
    args = parser.parse_args()

    players = []

    Game = None
    if args.game:
        module_path = '.'.join(args.game.split('.')[:-1])
        module = importlib.import_module(module_path)
        Game = getattr(module, args.game.split('.')[-1])

    for i in (0, 1):
        player_args = getattr(args, f'player{i}')
        playerstr = player_args[0]
        kwargs = {}
        for kwarg in player_args[1:]:
            k, s = kwarg.split('=')
            if s == 'True':
                v = True
            if s == 'False':
                v = False
            try:
                v = float(s)
            except:
                pass
            try:
                v = int(s)
            except:
                pass
            kwargs[k] = v

        if playerstr is None:
            player = HumanPlayer()
        elif playerstr == 'solver':
            player = SolverPlayer(**kwargs)
        else:
            model, config = load_ckpt(playerstr)
            assert Game is None or type(Game()) == type(config.Game()), 'Players must agree on game'
            Game = config.Game
            player = AlphaZeroPlayer(model, **kwargs)
        players.append(player)

    assert Game is not None, 'Game must be specified either in config or on the command line.'

    play(Game(), players, show=args.show)
