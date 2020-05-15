import numpy as np
import torch # TODO: Remove dependency here

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
        self.exploration = 0
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
        run_mcts(self._game, self._tree, self._model, 128, epsilon=0)
        if self._debug:
            N = self._tree.num_visits
            P = self._tree.action_prior
            print(f'Num visits ({np.sum(N)}): {N}')
            print(f'Action Prior: {P}')
        if self.exploration > 0:
            action_index = self._tree.sample_action(self.exploration)
        else:
            action_index = self._tree.greedy_action()
        self._tree = self._tree.traverse(action_index)
        self._tree.kill_siblings()
        action = self._game.valid_actions[action_index]
        return action, None

    def observe_action(self, action):
        action_index = self._game.valid_actions.index(action)
        self._tree = self._tree.traverse(action_index)
        self._tree.kill_siblings()


class SolverPlayer(Player):
    def __init__(self, temperature=None, debug=False):
        super(SolverPlayer, self).__init__()
        self._game = None
        assert temperature is None or temperature > 0
        self._temperature = temperature
        self._debug = debug

    def set_game(self, game):
        self._game = game

    def get_action(self):
        scores, _ = self._game.solve()
        valid_actions = self._game.valid_actions
        if self._debug:
            print('Move scores:', scores)
        if self._temperature is None:
            best_score = np.max(scores)
            indices = np.argwhere(scores == best_score)
            i = np.random.choice(indices.reshape(-1))
            action = valid_actions[i]
        else:
            scores = torch.tensor(scores, dtype=torch.float32)
            p = torch.nn.functional.softmax(scores / self._temperature, dim=0).numpy()
            action = np.random.choice(valid_actions, p=p)
            if self._debug:
                print('Sample likelihood:', p)
        return action, None


def load_agent_from_args(name, kwargs):

    Agent = {
        'Human': HumanPlayer,
        'Alphazero': AlphaZeroPlayer,
        'Solver': SolverPlayer,
    }.get(name)


