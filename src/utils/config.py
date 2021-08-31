# https://github.com/JiahuiYu/neuralgym/blob/master/neuralgym/utils/config.py

"""config utilities for yml file."""
import logging
import yaml

logger = logging.getLogger()


# get configs
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)


if __name__ == '__main__':
    config_path = '../config/config.yaml'
    conf = get_config(config_path)
    print(conf['batch_size'])  # with function
