import os
from alphazero.config import AlphaZeroConfig
from alphazero.game import Game

HERE = os.path.dirname(os.path.abspath(__file__))
DUMMY_CONFIG_PATH = os.path.join(HERE, 'configs/dummyconf.yaml')

class DummyGame(Game):
    NUM_ACTIONS = 4
    def __init__(self, history=None, num_players=2, max_turns=-1):
        super(DummyGame, self).__init__(history)
        self.num_players = num_players
        self.max_turns = max_turns

class DummyModel:
    def __init__(self, num_blocks=4, channels_per_block=16, some_setting=False):
        self.num_blocks = num_blocks
        self.channels_per_block = channels_per_block
        self.some_setting = some_setting

class DummyLoss:
    pass

def test_config():
    config = AlphaZeroConfig(DUMMY_CONFIG_PATH)

    # Class loading with args and kwargs
    game = config.Game()
    assert game.num_players == 4
    assert game.max_turns == 20
    assert game.history == []
    h = [0, 1, 2]
    game_w_history = config.Game(history=h)
    assert game_w_history.num_players == 4
    assert game_w_history.max_turns == 20
    assert game_w_history.history == h

    model = config.Model()
    assert model.num_blocks == 64
    assert model.channels_per_block == 256
    assert model.some_setting == True

    loss = config.Loss()

    # Selfplay settings
    assert config.selfplay['num_expansions_per_sim'] == 800
    assert config.selfplay['exploration_depth'] == 10

    # Training settings
    assert config.training['num_steps'] == 10000
    assert config.training['batch_size'] == 64
    assert config.training['lr'] == 0.02
    assert config.training['lr_schedule_gamma'] == 0.1
    assert config.training['lr_schedule'] == [500, 2000, 5000]
