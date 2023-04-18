import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import argparse
from easydict import EasyDict as edict

from tki.tools.utils import get_yml_content
from tki.controller.utils import check_args_from_input_config

from tki.controller.base_controller import BaseController
from tki.controller.dist_controller import DistController
from tki.controller.multi_controller import MultiController

def args_parser():
    parser = argparse.ArgumentParser('autosparsedl_config')
    parser.add_argument(
        '--config',
        type=str,
        default='./scripts/configs/cifar10/tki.yaml',
        help='yaml config file path')
    args = parser.parse_args()
    return args
    

def main():
    cmd_args = args_parser()
    yaml_configs = get_yml_content(cmd_args.config)
    yaml_configs = check_args_from_input_config(yaml_configs)
    yaml_configs = edict(yaml_configs)
    
    if yaml_configs.experiment.context.get('multi-p'):
        ctr = MultiController(yaml_configs)
    # elif yaml_configs.experiment.context.get('dist'):
    #     ctr = DistController(yaml_configs)
    else:
        ctr = BaseController(yaml_configs)
    
    ctr.run()
    
if __name__ == '__main__':
    main()

