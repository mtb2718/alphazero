from copy import deepcopy


# TODO: ABC?
class Game:

    NUM_ACTIONS = -1

    def __init__(self, history=None, search_statistics=None):
        assert self.NUM_ACTIONS > 0, 'Game classes must override NUM_ACTIONS class attribute.'
        self.history = history or []
        self.search_statistics = search_statistics or [None] * len(self.history)

    def __len__(self):
        return len(self.history)

    def __str__(self):
        raise NotImplementedError

    @property
    def next_player(self):
        return len(self.history) % 2

    @property
    def prev_player(self):
        return 1 - self.next_player

    def take_action(self, action, search_statistics=None):
        assert action in self.valid_actions, f'Cannot take invalid action "{action}"'
        self.history.append(action)
        self.search_statistics.append(search_statistics)

    def clone(self):
        return deepcopy(self)

    @property
    def terminal(self):
        raise NotImplementedError

    def terminal_value(self, player):
        winner = self.winner
        assert winner is not None
        if winner < 0:
            return 0
        return +1 if player == winner else -1

    @property
    def winner(self):
        raise NotImplementedError

    @property
    def valid_actions(self):
        raise NotImplementedError

    @property
    def action_names(self):
        return [str(a) for a in self.valid_actions]


    def render(self, turn=None):
        raise NotImplementedError
