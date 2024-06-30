import torch
import torch.nn as nn
from tabulate import tabulate
import torch.nn.functional as F

def print_model_summary(model):
    # get model information list
    params_info = []
    total_params = 0
    
    for name, param in model.named_parameters():
        layer_info = [name, list(param.shape), param.numel()]
        params_info.append(layer_info)
        total_params += param.numel()

    # add the count of total model parameters
    total_params_m = total_params / 1_000_000
    params_info.append(['Total', '-', f"{total_params} ({total_params_m:.2f}M)"])

    # print the whole table
    headers = ["Layer", "Shape", "Param #"]
    print(tabulate(params_info, headers=headers, tablefmt="grid"))
    
    
# calculate the relative error about the predict value and target value
def calculate_mre(pred_y, test_y):
    # calculate batch error
    return sum(((pred_y[i]-test_y[i])).flatten().norm().item() / test_y[i].flatten().norm().item() for i in range(pred_y.shape[0]))


# calculate the mae metric for the test data
def calculate_mae(pred_y, test_y):
    return F.l1_loss(pred_y, test_y).item() * pred_y.shape[0]
  
    
# calculate the max_mae metric for the test data
def calculate_max_mae(pred_y, test_y):
    return torch.sum(torch.max(torch.abs(test_y - pred_y).flatten(1), dim=1)[0]).item()
