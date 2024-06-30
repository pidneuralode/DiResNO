# validate and record corresponding data for the cylinder
import math
import torch
import torch.nn as nn
import csv
import torch.utils
import torch.utils.data
from tqdm import tqdm
from timeit import default_timer
import sys
import torch.nn.functional as F
from dataset.Training_data_v5 import TrainingDataModuleV5
from models.respace import SpacedDiffusion
from utils.common import calculate_mre
from utils.loss import MRE
import numpy as np
    
@torch.no_grad()
def evalu(residual_model: nn.Module, 
          diffusion_models: SpacedDiffusion, 
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler, 
          test_loader: torch.utils.data.DataLoader,
          config, device, 
          epoch: int, 
          data_module: TrainingDataModuleV5,
          loss_fn
          ):
    print('evaluation of the whole process')
    normalizer_x, normalizer_y = data_module.normalizer_x, data_module.normalizer_y
    test_mre, original_mre = 0.0, 0.0
    dsize = 0
    for test_inputs, test_labels, mark_v in test_loader:
        # infer the sample process
        dsize += test_inputs.shape[0]
        test_inputs = normalizer_x.encode(test_inputs).to(device)
        test_labels = normalizer_x.encode(test_labels).to(device)
        mark_v = mark_v.to(device)
        predict_labels = diffusion_models.p_sample_loop(test_inputs, residual_model)
        print(f'current batch predict shape:{predict_labels.shape}')
        # compute the mre loss between predict value and the ground truth
        test_mre += calculate_mre(predict_labels * mark_v, test_labels * mark_v)
        original_mre += calculate_mre(test_inputs * mark_v, test_labels * mark_v)
    print('------------------------------------------')
    test_mre /= dsize
    print(f'the test relative mean error is {test_mre}')
    print(f'the original relative mean error is {original_mre}')

def train(residual_model: nn.Module, 
          diffusion_models: SpacedDiffusion, 
          optimizer: torch.optim.Optimizer, 
          scheduler: torch.optim.lr_scheduler, 
          data_module: TrainingDataModuleV5, 
          config,
          device):
    # here we will train the resshift model for the whole process
    # convert resiual model to train mode
    train_loader, test_loader = data_module.train_dataloader(), data_module.test_dataloader()
    normalizer_x, normalizer_y = data_module.normalizer_x, data_module.normalizer_y
    # loss function for performance metrics
    loss_fn = MRE()
    # train within the max_iters
    for iter in range(int(config['train']['iterations'])):
        residual_model.train()
        loss_tmp = []
        for train_inputs, train_labels, mask_t in tqdm(train_loader):
            # begin train the residual model in batch process
            # encode the data
            train_inputs = normalizer_x.encode(train_inputs)
            train_labels = normalizer_x.encode(train_labels)
            current_batchsize = train_inputs.shape[0]
            micro_batchsize = int(config['train']['microbatch'])
            num_grad_accumulate = math.ceil(current_batchsize / micro_batchsize)

            optimizer.zero_grad()
            # residual diffuision in process, here we will construct different time steps 
            for mini_batch in range(0, current_batchsize, micro_batchsize):
                micro_train_inputs, micro_train_labels = train_inputs[mini_batch:min(mini_batch + micro_batchsize, current_batchsize),].to(device), \
                train_labels[mini_batch:min(mini_batch + micro_batchsize, current_batchsize),].to(device)
                # construct timestep for training 
                tt = torch.randint(
                    0, diffusion_models.num_timesteps,
                    size=(micro_train_inputs.shape[0],),
                    device=device,
                )
                # construct the noise for training
                noise = torch.randn(
                    size=micro_train_labels.shape,
                    device=device,
                ) 
                # compute the loss in the traning of resshift
                losses, z_t, z0_pred = diffusion_models.training_losses(residual_model, 
                                                                        micro_train_labels,
                                                                        micro_train_inputs, 
                                                                    tt,
                                                                    noise=noise,)
                loss = losses["loss"].mean() / num_grad_accumulate
                loss_tmp.append(loss.cpu().detach().numpy())
                loss.backward()
            # update model parameters
            optimizer.step()
        
        print('------------------------------------------')
        print(f'the train loss is {np.mean(loss_tmp)}')
        # adjust the lr
        scheduler.step()

        # record the model performance in test data
        if iter % 1 == 0:
            print('residual diffusion model test')
            residual_model.eval()
            evalu(residual_model, 
                  diffusion_models, 
                  optimizer, scheduler, 
                  test_loader, 
                  config, device, 
                  iter, 
                  data_module, 
                  loss_fn
                  )
        
        # save the model parameters
        if iter % 1000 == 0:
            torch.save({
                'epoch': iter,
                'model_state_dict': residual_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'lr_state_dict': scheduler.state_dict(),
            }, '/home/wangpengkai/node/DiResNO/logs/model_DiResNO.pth')


    
if __name__ == '__main__':
    # test mre function
    y1 = torch.Tensor([[1, 2, 3], [1, 2, 3]])
    y2 = torch.Tensor([[2, 2, 4], [2, 2, 4]])
    print(calculate_mre(y1, y2))
    print('test')






