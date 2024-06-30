import os
import numpy as np
from pathlib import Path
from typing import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from einops import repeat,rearrange

class ImageGaussianNormalizer(nn.Module):
    def __init__(self, mean, std, eps=1e-05):
        super().__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.eps = torch.tensor(eps)

    def encode(self, x):
        device = x.device
        x = (x - self.mean[None, :, None, None].to(device)) / (self.std[None, :, None, None] + self.eps).to(device)
        return x

    def decode(self, x):
        device = x.device
        x = x * (self.std[None, :, None, None] + self.eps).to(device) + self.mean[None, :, None, None].to(device)
        return x

    def to(self,device):
        self.mean = self.mean.to(device)
        self.std  = self.std .to(device)
        self.eps  = self.eps .to(device)


class TrainingDataModuleV5():
    def __init__(self, 
                 data_dir: str = '/home/wangpengkai/node/DiResNO/data', 
                 batch_size: int = 32, 
                 ):
        
        super().__init__()
        self.data_dir = data_dir
        self.train_data_dir = os.path.join(data_dir, 'train')
        self.test_data_dir = os.path.join(data_dir, 'test')
        self.mask_data_idr = os.path.join(data_dir, 'v5_mask.npy')
        self.vtk_img_map_dir = os.path.join(data_dir, 'v5_vtk_img_map.npy')
        self.batch_size = batch_size
    
    def convertVtk2Img(self, vtk_data):
        # vtk_data:(batch_size, number of points) here the channel of input data is 1 in default config
        v5_vtk_img_map = np.load(self.vtk_img_map_dir)
        print(v5_vtk_img_map.shape)
        # the shape will be (batch_size, H, W, in_channel=1)
        # TODO:here we clip, but we should do it in data-process
        return vtk_data[:, v5_vtk_img_map, np.newaxis][:,:80,:80,:]

    def load_data_into_tensors(self):
        self.train_features = self.convertVtk2Img(np.load(os.path.join(self.train_data_dir,'train_predict_z.npy')))
        self.train_labels   = self.convertVtk2Img(np.load(os.path.join(self.train_data_dir,'train_truth_z.npy')))
        self.test_features  = self.convertVtk2Img(np.load(os.path.join(self.test_data_dir,'test_predict_z.npy')))
        self.test_labels    = self.convertVtk2Img(np.load(os.path.join(self.test_data_dir,'test_truth_z.npy')))
        self.mask           = np.load(self.mask_data_idr)[:80, :80]
        
        self.train_features = torch.from_numpy(self.train_features).float()
        self.train_labels   = torch.from_numpy(self.train_labels  ).float()
        self.test_features   = torch.from_numpy(self.test_features  ).float()
        self.test_labels     = torch.from_numpy(self.test_labels    ).float()
        
        
        self.train_features = rearrange(self.train_features,'b x y c -> b c x y').float()
        self.train_labels   = rearrange(self.train_labels  ,'b x y c -> b c x y').float()
        self.test_features   = rearrange(self.test_features  ,'b x y c -> b c x y').float()
        self.test_labels     = rearrange(self.test_labels    ,'b x y c -> b c x y').float()
        
        self.mask           = torch.from_numpy(self.mask)
          
    def setup(self, stage: Optional[str] = None):
        self.load_data_into_tensors()
    
        print("self.train_features.shape", self.train_features.shape)
        print("self.train_labels.shape", self.train_labels.shape)
        print("self.test_features.shape", self.test_features.shape)
        print("self.test_labels.shape", self.test_labels.shape)

        mask_t = repeat(self.mask ,'x y -> b x y', b=len(self.train_labels))
        self.train_dataset = TensorDataset(self.train_features, self.train_labels, mask_t)
        mask_v = repeat(self.mask ,'x y -> b x y', b=len(self.test_labels))
        self.test_dataset = TensorDataset(self.test_features, self.test_labels, mask_v)

        self.setup_normalizer()
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,num_workers=4)

    def teardown(self, stage: Optional[str] = None):
        pass
    
    def setup_normalizer(self):
        mean_x = torch.mean(self.train_features, dim=[0, 2, 3])
        std_x  = torch.std(self.train_features, dim=[0, 2, 3])
        mean_y = torch.mean(self.train_labels, dim=[0, 2, 3])
        std_y  = torch.std(self.train_labels, dim=[0, 2, 3])
        print('mean_x ',mean_x )
        print('std_x  ',std_x  )
        print('mean_y ',mean_y )
        print('std_y  ',std_y  )
        self.data_normalizer_x = ImageGaussianNormalizer(mean_x,std_x)
        self.data_normalizer_y = ImageGaussianNormalizer(mean_y,std_y)
        self.normalizer_x = self.data_normalizer_x
        self.normalizer_y = self.data_normalizer_y


if __name__ == '__main__':
    data_module = TrainingDataModuleV5()
    data_module.setup()
    train_loader = data_module.train_dataloader()
    for x, y, mask_t in train_loader:
        print(x.shape)
        print(y.shape)
        print(mask_t.shape)
    test_loader = data_module.test_dataloader()
    for x, y, mask_v in test_loader:
        print(x.shape)
        print(y.shape)
        print(mask_v.shape)
