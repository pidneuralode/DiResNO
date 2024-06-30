import argparse
import numpy as np
import torch.cuda
from dataset.Training_data_v5 import TrainingDataModuleV5
from models.original_unet import UNet
from models.respace import SpacedDiffusion
import sys
from utils.optim import *
from utils.trainer import train
from utils.common import print_model_summary
from models_init_script import create_gaussian_diffusion
import yaml

parser = argparse.ArgumentParser('diffusion model as residual learner for car dataset')

# parse the argument from the terminal
parser.add_argument('--gpu', default=0, help='choice of gpu')
parser.add_argument('--config-path', default='/home/wangpengkai/node/DiResNO/config/default.yaml', help='the config file of this process')
# resolve the args
args = parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    # fix random seed
    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # set device
    device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device('cpu')

    # load config
    config = load_config(args.config_path)

    # residual model
    residual_model = UNet().to(device=device)
    print_model_summary(residual_model)

    # diffusoin model
    diffusion_models = create_gaussian_diffusion(**config['diffusion']['params'])

    # load data and the nomalizer
    data_module = TrainingDataModuleV5()
    data_module.setup()
    train_loader, test_loader = data_module.train_dataloader(), data_module.test_dataloader()
    print(f'train sample length:{len(train_loader)}, test sample length:{len(test_loader)}')

    # optimizer
    optimizer = instantiate_optimizer(residual_model, config)

    # scheduler
    scheduler = instantiate_scheduler(optimizer, config)

    # train the whole process in car dataset
    train(residual_model, 
          diffusion_models,
          optimizer, 
          scheduler,
          data_module,
          config,
          device,
          )


if __name__ == '__main__':
    main()