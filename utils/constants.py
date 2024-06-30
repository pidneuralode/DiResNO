from torch import nn
import torch.nn.functional as F


ACTIVATION = {'gelu':nn.GELU(),
              'tanh':nn.Tanh(),
              'sigmoid':nn.Sigmoid(),
              'relu':nn.ReLU(),
              'leaky_relu':nn.LeakyReLU(0.1),
              'softplus':nn.Softplus(),
              'ELU':nn.ELU()}
