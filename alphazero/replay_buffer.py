from itertools import repeat
import os
import sqlite3

import numpy as np
from torch.utils.data import Dataset, Sampler


class SelfPlayDatabase(sqlite3.Connection):

    @staticmethod
    def connect(dbpath):
        return sqlite3.connect(dbpath, factory=SelfPlayDatabase)

    def init_db(self):
        self.execute("""CREATE TABLE IF NOT EXISTS selfplay_games (
            id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            model_version INTEGER NOT NULL,
            history TEXT NOT NULL,
            search_statistics BLOB NOT NULL);""")

        self.execute("""CREATE TABLE IF NOT EXISTS game_states (
            id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            game_id INTEGER NOT NULL,
            move_index INTEGER NOT NULL);""")
        self.commit()

    def write_game(self, game, model_version):

        history = ','.join([str(a) for a in game.history])
        assert len(game.search_statistics) == len(game.history)
        assert game.terminal
        assert game.search_statistics[0] is not None
        search_statistics = np.concatenate(game.search_statistics)
        search_statistics_blob = search_statistics.astype(np.int16).tostring()

        # Insert game into selfplay_games table and a row for each (non-terminal) state in the game
        # into the game_states table. Note that these writes are atomic; all other write requests
        # from other connections will block until we .commit() these ones.
        cur = self.execute("""INSERT INTO selfplay_games
            (model_version, history, search_statistics) VALUES (?, ?, ?);""",
            (model_version, history, search_statistics_blob))
        game_id = cur.lastrowid
        self.executemany("""INSERT INTO game_states
            (game_id, move_index) VALUES (?, ?);""",
            zip(repeat(game_id), range(len(game.history))))
        self.commit()

    def read_game(self, i, game):
        assert len(game.history) == 0
        game_id = i + 1
        cur = self.execute("""SELECT model_version, history, search_statistics
            FROM selfplay_games WHERE id = ?;""", (game_id,))
        row = cur.fetchone()
        if row is None:
            raise ValueError
        model_version, history, search_statistics_blob = row
        history = [int(a) for a in history.split(',')]
        search_stats = np.frombuffer(search_statistics_blob, dtype=np.int16)
        n = 0
        for action in history:
            A = len(game.valid_actions)
            game.take_action(action, search_stats[n:n + A])
            n += A
        return game, model_version

    def read_state(self, i):
        state_id = i + 1
        cur = self.execute("""SELECT game_id, move_index
            FROM game_states WHERE id = ?;""", (state_id,))
        row = cur.fetchone()
        if row is None:
            raise ValueError
        game_id, move_index = row
        return game_id - 1, move_index

    def num_games(self):
        cur = self.execute("SELECT COUNT(*) FROM selfplay_games;")
        return cur.fetchone()[0]

    def num_states(self):
        cur = self.execute("SELECT COUNT(*) FROM game_states;")
        return cur.fetchone()[0]


def make_forksafe(method):
    """Decorator for RepalyBufferDataset methods that make use of a database
    connection. Since SQLite connections aren't fork-safe, we must ensure
    that a forked connection is not used to avoid database corruption.
    # See: https://www.sqlite.org/howtocorrupt.html#fork
    """
    def forksafe_wrapper(self, *args, **kwargs):
        pid = os.getpid()
        if pid != self._pid:
            self._db = SelfPlayDatabase.connect(self._dbpath)
            self._pid = pid
        return method(self, *args, **kwargs)
        return result
    return forksafe_wrapper


class ReplayBufferDataset(Dataset):

    def __init__(self, config, dbpath):
        self._config = config
        self._dbpath = dbpath
        self._db = SelfPlayDatabase.connect(self._dbpath)
        self._db.init_db()
        self._pid = os.getpid()

    @make_forksafe
    def __getitem__(self, i):
        # Retrieve the game and move index referenced by the i'th game state
        game_index, move_index = self._db.read_state(i)
        finished_game, model_version = self._db.read_game(game_index, self._config.Game())

        # Consider the intermediate state of the game after some moves
        history = finished_game.history[:move_index]
        game = self._config.Game(history=history)

        # Render the game state for neural network consumption
        x = game.render()

        # Construct target for action prior from search statistics
        n = finished_game.search_statistics[move_index]
        p = np.zeros(game.NUM_ACTIONS, np.float32)
        p[game.valid_actions] = n / np.sum(n)

        # Mask of valid actions
        m = np.zeros(game.NUM_ACTIONS, np.bool)
        m[game.valid_actions] = True

        # Target for the value of the game state is the final outcome of the
        # game, from the perspective of the next player to move.
        next_player = move_index % 2
        z = np.array([finished_game.terminal_value(next_player)], dtype=np.float32)

        return {
            'x': x,
            'z': z,
            'p': p,
            'p_valid': m,
            'game_index': game_index,
            'move_index': move_index,
            'model_version': model_version,
        }

    @make_forksafe
    def __len__(self):
        return self._db.num_states()

    @make_forksafe
    def add_game(self, game, model_version):
        assert game.terminal
        self._db.write_game(game, model_version)


class ReplayBufferSampler(Sampler):
    def __init__(self, dataset, batch_size, bufferlen=0):
        # TODO: Configurable support for exponential/curriculum sampling
        self.dataset = dataset
        self.batch_size = batch_size
        self.bufferlen = bufferlen

    def __len__(self):
        return 1e9

    def __iter__(self):
        return self

    def __next__(self):
        hi = len(self.dataset)
        lo = max(hi - self.bufferlen, 0) if self.bufferlen > 0 else 0
        if hi == 0:
            raise StopIteration
        replace = self.batch_size > hi - lo
        indices = lo + np.random.choice(hi - lo, self.batch_size, replace)
        return indices.tolist()
