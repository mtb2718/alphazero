# AlphaZero Implementation

## Environment Setup

### Create a new conda environment:

```bash
conda create -n alphazero python=3.7
conda activate alphazero
```

### Install PyTorch

See [PyTorch website](https://pytorch.org/get-started/locally/#start-locally)
for latest & most appropriate command for your setup.

On a laptop with no GPU, for instance, you may try the following
command:
```bash
conda install pytorch torchvision cpuonly -c pytorch
```
This setup is only practical for training and experimenting in
toy-problem settings, like tic-tac-toe.

### Install other python dependencies via pip

```bash
pip install -r requirements.txt
```

### Install Redis

Redis is an in-memory datastore used to sync the latest models
during training with distributed self-play workers.
Note that this dependency is not necessary for single-process
training in toy-problem settings. (However, it is still currently
_required_. TODO: make Redis dependency optional.)

Build from source and install (recommended):

```bash
wget http://download.redis.io/redis-stable.tar.gz
tar xvzf redis-stable.tar.gz
cd redis-stable
make
```

You may now launch the `src/redis-server` binary in another tab
before starting training, or alternatively follow the instructions
in the [Redis quickstart guide](https://redis.io/topics/quickstart)
to finish by installing and daemonizing the server (recommended).

### Install Connect4 Solver (Required for Connect4 Evaluation)

Tic-tac-toe and Connect4 are strongly solved games, meaning that for any board position
the best possible move(s) and the relative strength of each possible move are known
a priori. As such, these games make for especially enlightening testbeds, as we can
accurately evaluate both the predictions from our neural network models and the quality
of the training targets produced by self-play. Solving tic-tac-toe is trivial, included
in tictactoe.py in < 50 lines of code. In Connect4, quite a bit more effort is required to solve positions
efficiently, so we rely on an existing open-source solver by Pascal Pons.

Clone and build the [Connect4 solver](https://github.com/PascalPons/connect4).

```bash
git clone https://github.com/PascalPons/connect4.git
cd connect4
make c4solver
```

The solver optionally makes use of a "book" of precomputed solutions, which decreases solve
times by orders of magnitude.
[Download the book of precomputed solutions](https://github.com/PascalPons/connect4/releases/download/book/7x6.book).

Our ConnectFour class's `solve()` method assumes both the solver and book are "installed" in your system:

```bash
sudo cp c4solver /usr/local/bin/
sudo cp 7x6.book /usr/local/bin/
```

### Run Tests

Finally, confirm that everything is working as expected by running the included unit tests:

```bash
pytest
```