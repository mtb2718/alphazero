from argparse import ArgumentParser

import numpy as np
import torch

from connectfour import AlphaZeroC4, ConnectFourState, GRID_WIDTH
from mcts import MCTreeNode


class Player:
    # TODO: ABC
    def __init__(self):
        pass
    def get_action(self):
        pass


class HumanPlayer(Player):
    def __init__(self):
        super(HumanPlayer, self).__init__()

    def get_action(self, state):
        print(state)
        a = input(f'(Player {state.turn}) Take Action [0-{GRID_WIDTH - 1}, (u)ndo, (q)uit], (d)ebug: ')
        try:
            action = int(a)
            action_index = state.valid_actions.index(action)
            return action_index, None
        except:
            return None, a


class AlphaZeroPlayer(Player):
    def __init__(self, model, debug=False):
        super(AlphaZeroPlayer, self).__init__()
        self._model = model
        self._debug = debug

    def get_action(self, state):
        # TODO: Could keep accumulating expanded nodes throughout game
        # This should make AI much stronger
        if self._debug:
            print(state)
        node = MCTreeNode(state)
        with torch.no_grad():
            for _ in range(1024):
                node.expand(self._model)
        action_index = np.argmax(node.pi())
        return action_index, None


def play(state, players, current_player=0):
    """Play game from given state with given players."""

    states = [state]
    while True:

        player = players[current_player]
        action_index, game_control = player.get_action(state)

        if game_control is None:
            assert action_index >= 0 and action_index < len(state.valid_actions)
            state = state.copy()
            state.take_action(action_index)
            states.append(state)
            current_player = (current_player + 1) % len(players)
        elif game_control == 'q':
            exit(0)
        elif game_control == 'u':
            if len(states) > 1:
                state = states[-2]
                states = states[:-1]
                current_player = (current_player - 1) % len(players)
        elif game_control == 'd':
            import pdb
            pdb.set_trace()
            continue
        else:
            print('Invalid Selection.')
            continue

        if state.winner is None:
            continue
        elif state.winner >= 0:
            print(state)
            print(f'Player {state.winner} wins!')
        else:
            print(state)
            print('Game drawn!')
        break


def load_ckpt(ckpt):
    model = AlphaZeroC4()
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('ckpt',
        help="Path to a ckpt.pt file")
    # TODO: Support N-player games with arbitrary arrangments of human and ckpt players
    '''
    parser.add_argument('-p', '--player',
        type=int,
        default=0,
        help="Select index of human player.")
    '''
    args = parser.parse_args()

    model = load_ckpt(args.ckpt)
    play(ConnectFourState(), [
        AlphaZeroPlayer(model, True),
        #AlphaZeroPlayer(model, True),
        HumanPlayer(),
    ])
