import _init_path
import os
import pprint
import argparse
import numpy as np
import torch
import importlib

from common.utils.config import parse_args
from common.utils.logger import create_logger, summary_parameters
from common.utils.data import make_dataloader
from common.metric import *
from modules import ExpNet

def main():
    config = parse_args()

    logger, final_output_path = create_logger(config.OUTPUT_PATH, config.MODEL_PREFIX)
    logger.info('training config:{}\n'.format(pprint.pformat(config)))

    if config.RNG_SEED > -1:
        np.random.seed(config.RNG_SEED)
        torch.random.manual_seed(config.RNG_SEED)
        torch.cuda.manual_seed_all(config.RNG_SEED)
    
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = True

    model = eval(config.NETWORK.BACKBONE)(config)
    summary_parameters(model, logger)
    num_gpus = len(config.GPUS.split(','))
    if num_gpus > 1:
        model = torch.nn.DataParallel(
            model, device_ids=[int(d) for d in config.GPUS.split(',')]
        ).cuda()
    else:
        torch.cuda.set_device(int(config.GPUS))
        model.cuda()
    
    train_loader = make_dataloader(config, 'train')
    val_loader = make_dataloader(config, 'val')

    batch_size = num_gpus * config.TRAIN.BATCH_SIZE * config.TRAIN.GRAD_ACCUMULATE_STEPS
    base_lr = config.TRAIN.LR * batch_size
    optimizer_params = [{
        'params': [p for n, p in model.named_parameters()]
    }]
    
    if config.TRAIN.OPTIMIZER == 'SGD':
        optimizer = torch.optim.SGD(
            optimizer_params, lr = base_lr, 
            momentum = config.TRAIN.MOMENTUM,
            weight_decay= config.TRAIN.WD
        )
    elif config.TRAIN.OPTIMIZER == 'Adam':
        optimizer = torch.optim.Adam(
            optimizer_params, lr = base_lr,
            weight_decay = config.TRAIN.WD
        )
    else:
        raise ValueError('Not supported optimizer {}!'.format(config.TRAIN.OPTIMIZER))
    
    MetricClass = importlib.import_module('modules.{}.metric'.format(config.TASK)).Metric(config)
    train_metrics = MetricClass()
    val_metrics = MetricClass()

if __name__ == '__main__':
    main()