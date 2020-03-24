# AlphaZero Implementation

## Environment Setup

### Create a new conda environment:

```bash
conda create -n alphazero python=3.7
source activate alphazero
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

### Run Tests

Finally, confirm that everything is working as expected by running included unit tests:

```bash
pytest
```