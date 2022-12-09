import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils import get_shuffled_idx

class JetsIMG(Dataset):
    def __init__(self,dataframe_path, split='training', size=40, key='table', seed=1, transform=None):
        ##########implement sample size selection##########
        df = pd.read_hdf(dataframe_path, key=key)
        
        assert split in ('training', 'validation', 'evaluation')
        if split == 'training':
            df = df[:100000]
            shuffle_idx = get_shuffled_idx(len(df), seed)
            is_sign = df.pop('is_signal_new')
            mass = df.pop('mass')
            
            self.data = df.iloc[:, :1600].to_numpy().reshape(-1, size, size, 1)[shuffle_idx]
            self.is_sign = is_sign.to_numpy()[shuffle_idx]
            self.mass = mass.to_numpy()[shuffle_idx]
        
        elif split == 'validation':
            df = df[int(.6*len(df)):int(.8*len(df))]
            is_sign = df.pop('is_signal_new')
            mass = df.pop('mass')
            self.data = df.iloc[:, :1600].to_numpy().reshape(-1,size,size,1)
            self.is_sign = is_sign.to_numpy()
            self.mass = mass.to_numpy()
        
        else:
            df = df[int(.8*len(df)):]
            is_sign = df.pop('is_signal_new')
            mass = df.pop('mass')
            self.data = df.iloc[:, :1600].to_numpy().reshape(-1,size,size,1)
            self.is_sign = is_sign.to_numpy()
            self.mass = mass.to_numpy()

        self.data = self.data.astype('float32')
        self.is_sign = self.is_sign.astype('float32')
        self.mass = self.mass.astype('float32')
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, self.is_sign[idx]

