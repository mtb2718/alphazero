import importlib
import yaml

def _load_class(module_path, name):
    module = importlib.import_module(module_path)
    return getattr(module, name)

def _load_instance(class_conf, kwargs={}):
    Class = _load_class(class_conf['module'], class_conf['name'])
    kw = {**class_conf.get('kwargs', {}), **kwargs}
    return Class(**kw)


class AlphaZeroConfig:
    def __init__(self, path):
        self._config_path = path
        with open(path) as f:
            config = yaml.safe_load(f)
        self._conf = config

    def Game(self, **kwargs):
        return _load_instance(self._conf['game'], kwargs)

    def Model(self):
        return _load_instance(self._conf['model'])

    def Loss(self):
        return _load_instance(self._conf['loss'])

    def __getattr__(self, attr):
        # TODO: Could be nice to make a read-only nested dict view
        assert attr in ['selfplay', 'training']
        return self._conf[attr]
