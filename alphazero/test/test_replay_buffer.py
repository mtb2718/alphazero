import os

import numpy as np
import pytest
from torch.utils.data import DataLoader

from alphazero.config import AlphaZeroConfig
from alphazero.games.tictactoe import TicTacToe
from alphazero.replay_buffer import (
    ReplayBufferDataset,
    ReplayBufferSampler,
    SelfPlayDatabase)

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, 'configs/tictactoe.yaml')

def _play_game(history, num_expansions=128):
    game = TicTacToe()
    for action in history:
        L = len(game.valid_actions)
        n = np.random.randint(0, L, size=num_expansions)
        stats = np.array([np.sum(n == i) for i in range(L)], dtype=np.uint32)
        game.take_action(action, search_statistics=stats)
    return game


def test_database(tmp_path):
    dbpath = str(tmp_path / 'testdb.sqlite')
    db = SelfPlayDatabase.connect(dbpath)
    db.init_db()
    assert db.num_games() == 0

    GAME_HISTORY = [0, 2, 8, 4, 6, 7, 3]
    MODEL_VERSION = 42
    game = _play_game(GAME_HISTORY)
    db.write_game(game, MODEL_VERSION)

    assert db.num_games() == 1
    assert db.num_states() == len(GAME_HISTORY)

    game0, v0 = db.read_game(0, TicTacToe())
    assert game0 is not game
    assert v0 == 42
    assert game0.history == game.history
    for i, _ in enumerate(game.history):
        assert np.all(game0.search_statistics[i] == game.search_statistics[i])

    game_index, move_index = db.read_state(0)
    assert game_index == 0
    assert move_index == 0
    game_index, move_index = db.read_state(len(GAME_HISTORY) - 1)
    assert game_index == 0
    assert move_index == len(GAME_HISTORY) - 1
    assert db.num_states() == len(GAME_HISTORY)

    db.write_game(game, 43)
    assert db.num_games() == 2
    assert db.num_states() == 2 * len(GAME_HISTORY)
    game1, v1 = db.read_game(1, TicTacToe())
    assert v1 == 43
    assert game1.history == game0.history
    game_index, move_index = db.read_state(len(GAME_HISTORY) + 1)
    assert game_index == 1
    assert move_index == 1


def test_dataset(tmp_path):
    dbpath = str(tmp_path / 'testdb.sqlite')
    config = AlphaZeroConfig(CONFIG)

    GAME0_HISTORY = [0, 2, 8, 4, 6, 7, 3]
    GAME1_HISTORY = [0, 2, 8, 4, 6, 3, 7]
    MODEL_VERSION = 42

    game0 = _play_game(GAME0_HISTORY)
    game1 = _play_game(GAME1_HISTORY)

    dataset = ReplayBufferDataset(config, dbpath)
    assert len(dataset) == 0

    dataset.add_game(game0, MODEL_VERSION)
    assert len(dataset) == len(GAME0_HISTORY)
    for i in range(len(dataset)):
        example = dataset[i]
        assert example['model_version'] == MODEL_VERSION
        assert example['z'] == 1 - 2 * (i % 2)
        assert np.sum(example['p_valid']) == 9 - i
        assert np.sum(example['x']) == i
        assert np.sum(example['p']) == 1.

    dataset.add_game(game1, MODEL_VERSION)
    assert len(dataset) == len(GAME0_HISTORY) + len(GAME1_HISTORY)
    for i in range(len(GAME1_HISTORY)):
        j = len(GAME0_HISTORY) + i
        example = dataset[j]
        assert example['model_version'] == MODEL_VERSION
        assert example['z'] == 1 - 2 * (i % 2)
        assert np.sum(example['p_valid']) == 9 - i
        assert np.sum(example['x']) == i
        assert np.sum(example['p']) == 1.


def test_dataloader(tmp_path):
    dbpath = str(tmp_path / 'testdb.sqlite')
    config = AlphaZeroConfig(CONFIG)

    GAME_HISTORY = [0, 2, 8, 4, 6, 7, 3]
    game = _play_game(GAME_HISTORY)

    dataset = ReplayBufferDataset(config, dbpath)
    sampler = ReplayBufferSampler(dataset, batch_size=8)
    dataloader = DataLoader(dataset,
                            batch_sampler=sampler,
                            num_workers=0)
    assert len(dataset) == 0
    batch_iter = iter(dataloader)
    with pytest.raises(StopIteration):
        batch = next(batch_iter)

    for i in range(20):
        dataset.add_game(game, i)

    assert len(dataset) == 20 * len(GAME_HISTORY)
    batch_iter = iter(dataloader)
    batch = next(batch_iter)
    assert batch['x'].shape[0] == 8

    sampler.bufferlen = 8
    states_seen = set()
    for _ in range(100):
        batch = next(batch_iter)
        game_nums = batch['game_index'].tolist()
        move_nums = batch['move_index'].tolist()
        state_ids = list(zip(game_nums, move_nums))
        assert len(state_ids) == len(set(state_ids)), 'No dup examples in batch'
        states_seen |= set(state_ids)
    assert len(states_seen) == sampler.bufferlen

    sampler.bufferlen = 4
    batch = next(batch_iter)
    game_nums = batch['game_index'].tolist()
    move_nums = batch['move_index'].tolist()
    state_ids = list(zip(game_nums, move_nums))
    assert len(state_ids) > len(set(state_ids)), 'Batch must contain dups'


def test_dataloader_with_workers(tmp_path):
    dbpath = str(tmp_path / 'testdb.sqlite')
    config = AlphaZeroConfig(CONFIG)

    GAME_HISTORY = [0, 2, 8, 4, 6, 7, 3]
    game = _play_game(GAME_HISTORY)

    dataset = ReplayBufferDataset(config, dbpath)
    sampler = ReplayBufferSampler(dataset, batch_size=8)
    dataloader = DataLoader(dataset,
                            batch_sampler=sampler,
                            num_workers=4)
    assert len(dataset) == 0
    batch_iter = iter(dataloader)
    with pytest.raises(StopIteration):
        batch = next(batch_iter)

    for i in range(20):
        dataset.add_game(game, i)

    assert len(dataset) == 20 * len(GAME_HISTORY)
    batch_iter = iter(dataloader)
    batch = next(batch_iter)
    assert batch['x'].shape[0] == 8

    sampler.bufferlen = 8
    [next(batch_iter) for _ in range(16)] # Flush pre-loaded data
    states_seen = set()
    for _ in range(100):
        batch = next(batch_iter)
        game_nums = batch['game_index'].tolist()
        move_nums = batch['move_index'].tolist()
        state_ids = list(zip(game_nums, move_nums))
        assert len(state_ids) == len(set(state_ids)), 'No dup examples in batch'
        states_seen |= set(state_ids)
    assert len(states_seen) == sampler.bufferlen

    sampler.bufferlen = 4
    [next(batch_iter) for _ in range(16)] # Flush pre-loaded data
    batch = next(batch_iter)
    game_nums = batch['game_index'].tolist()
    move_nums = batch['move_index'].tolist()
    state_ids = list(zip(game_nums, move_nums))
    assert len(state_ids) > len(set(state_ids)), 'Batch must contain dups'
