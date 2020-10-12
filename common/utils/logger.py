import os
import errno
import logging
import time

def makedirsExist(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory not created.')
        else:
            raise

def create_logger(root_output_path, cfg_name):
    if not os.path.exists(root_output_path):
        makedirsExist(root_output_path)
    assert os.path.exists(root_output_path), '{} does not exist'.format(root_output_path)

    final_output_path = os.path.join(root_output_path, cfg_name)
    if not os.path.exists(final_output_path):
        makedirsExist(final_output_path)

    log_file = '{}_{}.log'.format(cfg_name, time.strftime('%Y-%m-%d-%H-%M'))
    logging.basicConfig(
        filename=os.path.join(final_output_path, log_file), 
        format='%(asctime)-15s %(message)s'
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    return logger, final_output_path

def summary_parameters(model, logger):
    logger.info('>> Trainable Parameters: ')
    trainable_paramters = [
        (str(n), str(v.dtype), str(tuple(v.shape)), str(v.numel()))
        for n, v in model.named_parameters() if v.requires_grad
    ]
    max_lens = [max([len(item) + 4 for item in col]) for col in zip(*trainable_paramters)]
    raw_format = '|' + '|'.join(['{{:{}s}}'.format(max_len) for max_len in max_lens]) + '|'
    raw_split = '-' * (sum(max_lens) + len(max_lens) + 1)
    logger.info(raw_split)
    logger.info(raw_format.format('Name', 'Dtype', 'Shape', '#Params'))
    logger.info(raw_split)
    for name, dtype, shape, number in trainable_paramters:
        logger.info(raw_format.format(name, dtype, shape, number))
    logger.info(raw_split)

    num_trainable_params = sum([v.numel() for v in model.parameters() if v.requires_grad])
    total_params = sum([v.numel() for v in model.parameters()])
    non_trainable_params = total_params - num_trainable_params
    logger.info('>> {:25s}\t{:.2f}\tM'.format('# TrainableParams:', num_trainable_params / (1.0 * 10 ** 6)))
    logger.info('>> {:25s}\t{:.2f}\tM'.format('# NonTrainableParams:', non_trainable_params / (1.0 * 10 ** 6)))
    logger.info('>> {:25s}\t{:.2f}\tM'.format('# TotalParams:', total_params / (1.0 * 10 ** 6)))
