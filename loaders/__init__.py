import numpy as np
import copy
import torch
from torch.utils import data
from torchvision.transforms import ToTensor, Compose

from loaders.hep_dataset import JetsIMG

from augmentations import get_composed_augmentations

def get_dataloader(data_dict, mode=None, mode_dict=None, data_aug=None):
    """constructs DataLoader
    data_dict: data part of cfg

    Example data_dict
        dataset: FashionMNISTpad_OOD
        path: datasets
        shuffle: True
        batch_size: 128
        n_workers: 8
        split: training
        dequant:
          UniformDequantize: {}
    """

    # dataset loading
    aug = get_composed_augmentations(data_dict.get('augmentations', None))
    dequant = get_composed_augmentations(data_dict.get('dequant', None))
    dataset = get_dataset(data_dict, split_type=None, data_aug=aug, dequant=dequant)

    # dataloader loading
    loader = data.DataLoader(
        dataset,
        batch_size=data_dict["batch_size"],
        num_workers=data_dict["n_workers"],
        shuffle=data_dict.get('shuffle', False),
        pin_memory=False,
    )

    return loader


def get_dataset(data_dict, split_type=None, data_aug=None, dequant=None):
    """
    split_type: deprecated argument
    """
    do_concat = any([k.startswith('concat') for k in data_dict.keys()])
    if do_concat:
        if data_aug is not None:
            return data.ConcatDataset([get_dataset(d, data_aug=data_aug) for k, d in data_dict.items() if k.startswith('concat')])
        elif dequant is not None:
            return data.ConcatDataset([get_dataset(d, dequant=dequant) for k, d in data_dict.items() if k.startswith('concat')])
        else: return data.ConcatDataset([get_dataset(d) for k, d in data_dict.items() if k.startswith('concat')])
    name = data_dict["dataset"]
    split_type = data_dict['split']
    data_path = data_dict["path"][split_type] if split_type in data_dict["path"] else data_dict["path"]

    # default tranform behavior. 
    original_data_aug = data_aug
    if data_aug is not None:
        #data_aug = Compose([data_aug, ToTensor()])
        data_aug = Compose([ToTensor(), data_aug])
    else:
        data_aug = ToTensor()

    if dequant is not None:  # dequantization should be applied last
        data_aug = Compose([data_aug, dequant])


    # datasets
    if name == 'JetsIMG':
        seed = data_dict.get('seed', 1)
        dataset = JetsIMG(data_path, split=split_type, seed=seed, transform=data_aug)
        
    else:
        raise NameError("Dataset not defined")

    return dataset

def np_to_loader(l_tensors, batch_size, num_workers, load_all=False, shuffle=False):
    '''Convert a list of numpy arrays to a torch.DataLoader'''
    if load_all:
        dataset = data.TensorDataset(*[torch.Tensor(X).cuda() for X in l_tensors])
        num_workers = 0
        return data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    else:
        dataset = data.TensorDataset(*[torch.Tensor(X) for X in l_tensors])
        return data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=False)
