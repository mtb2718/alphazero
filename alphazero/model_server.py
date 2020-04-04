import os
import pickle

from redis import Redis
import torch

class ModelServer:

    def __init__(self, logdir, host='localhost', port=6379, db=0, ckpt_period=50):
        self._ckpt_dir = os.path.join(logdir, 'ckpt')
        os.makedirs(self._ckpt_dir, exist_ok=True)
        self._ckpt_period = ckpt_period
        self._redis = Redis(host=host, port=port, db=db)

    def reset(self):
        self._redis.delete('model')
        self._redis.delete('version')

    def update(self, model, version):

        # Create a lock for this model and add it to our model cache
        model_data = pickle.dumps(model.state_dict())
        self._redis.mset({
            'model': model_data,
            'version': version,
        })

        # Periodically keep a checkpoint in the logdir
        if self._ckpt_period > 0 and version % self._ckpt_period == 0:
            self.checkpoint(self._ckpt_dir, model, version)

    def latest(self, model):

        vals = self._redis.mget(['model', 'version'])
        version = int(vals[1] or 0)
        if vals[0]:
            model_data = pickle.loads(vals[0])
            model.load_state_dict(model_data)

        return version

    def checkpoint(self, dirname, model, version):
        torch.save({
            'model_version': version,
            'model_state_dict': model.state_dict(),
            # TODO: Move checkpointing to an outside utility
            #'optimizer_state_dict': optimizer.state_dict(),
        }, f'{dirname}/{version:06d}.pt')
