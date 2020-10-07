from easydict import EasyDict as edict
import yaml

def load_config(config_file):
    with open(config_file) as f:
        cfg = edict(yaml.load(f))
        return cfg