from argparse import ArgumentParser

import numpy as np
import torch

from connectfour import AlphaZeroC4, ConnectFourState, GRID_WIDTH
from mcts import MCTreeNode


class Player:
    # TODO: ABC
    def __init__(self):
        pass
    def set_game_state(self, state):
        pass
    def get_action(self):
        pass
    def observe_action(self, action_index, new_state):
        pass


class HumanPlayer(Player):
    def __init__(self):
        super(HumanPlayer, self).__init__()
        self._state = None

    def set_game_state(self, state):
        self._state = state

    def get_action(self):
        print(self._state)
        a = input(f'(Player {self._state.turn}) Take Action [0-{GRID_WIDTH - 1}, (u)ndo, (q)uit], (d)ebug: ')
        try:
            action = int(a)
            action_index = self._state.valid_actions.index(action)
            return action_index, None
        except:
            return None, a

    def observe_action(self, action_index, new_state):
        self._state = new_state


class AlphaZeroPlayer(Player):
    def __init__(self, model, debug=False):
        super(AlphaZeroPlayer, self).__init__()
        self._model = model
        self._debug = debug
        self._search_tree = None

    def set_game_state(self, state):
        self._search_tree = MCTreeNode(state)

    def get_action(self):
        if self._debug:
            print(self._search_tree.state)
        with torch.no_grad():
            for _ in range(1024):
                self._search_tree.expand(self._model)
        if self._debug:
        action_index = np.argmax(self._search_tree.pi())
        self._search_tree = self._search_tree.traverse(action_index)
        self._search_tree.kill_siblings()
        return action_index, None

    def observe_action(self, action_index, new_state):
        self._search_tree = self._search_tree.traverse(action_index)
        self._search_tree.kill_siblings()


def play(state, players, current_player_index=0):
    """Play game from given state with given players."""

    states = [state]
    for player in players:
        player.set_game_state(state.copy())

    while True:

        player = players[current_player_index]
        action_index, game_control = player.get_action()

        if game_control is None:
            assert action_index >= 0 and action_index < len(state.valid_actions)
            state = state.copy()
            state.take_action(action_index)
            states.append(state)
            for p in players:
                if p != player:
                    p.observe_action(action_index, state.copy())
            current_player_index = (current_player_index + 1) % len(players)
        elif game_control == 'q':
            exit(0)
        elif game_control == 'u':
            if len(states) > 1:
                state = states[-2]
                states = states[:-1]
                current_player_index = (current_player_index - 1) % len(players)
        elif game_control == 'd':
            import pdb
            pdb.set_trace()
            continue
        else:
            print('Invalid Selection.')
            continue

        if state.winner is None:
            continue
        else:
            print(state)
            if state.winner >= 0:
                print(f'Player {state.winner} wins!')
            else:
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
