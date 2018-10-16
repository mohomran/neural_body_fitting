import os
import imp
import logging

LOGGER = logging.getLogger(__name__)


def get_config():
    CONF_FP = os.path.join(os.path.dirname(__file__), "options.py")
    LOGGER.info("Loading experiment configuration from `%s`...", CONF_FP)
    options = imp.load_source('_options', os.path.abspath(CONF_FP))
    
    dataset = options.config['dataset']
    CONF_FP = os.path.join("datasets", "metadata", dataset + ".py")
    LOGGER.info("Loading dataset configuration from `%s`...", CONF_FP)
    dataset = imp.load_source('_dataset', os.path.abspath(CONF_FP))
    
    options.config.update(dataset.config)
    LOGGER.info("Done.")
    
    return options.config


def adjust_config(config, mode):
    # Don't misuse this!
    config['mode'] = mode
    if mode not in ['train', 'trainval'] and not ['ignore_batchnorm_training_stats']:
        config['batch_size'] = 1
    return config
