from easydict import EasyDict as edict
import yaml
import argparse

def load_config(config_file):
    with open(config_file) as f:
        cfg = edict(yaml.load(f))
        return cfg

def parse_args():
    parser = argparse.ArgumentParser('Arguments')
    parser.add_argument('--config-file', type=str, help='Path to config.yaml')
    args = parser.parse_args()

    config = load_config(args.config_file)
    return config