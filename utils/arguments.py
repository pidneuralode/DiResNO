import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Process config')
    # here we just use the teminal information as input
    parser.add_argument('--config-path', default='/home/wangpengkai/node/ijcai-cfd/configs/V5CNN/U.yaml', type=str, help='Path to the config file')
    parser.add_argument('--use-tb', default=False,)
    parser.add_argument('--use-wandb', action='store_true', default=False)
    parser.add_argument('--wandb-log-key', type=str, default='26314e4524d6618fa12eb7c1a2d4e779f726a505')
    
    args = parser.parse_args()
    return args