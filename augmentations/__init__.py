import logging
from torchvision.transforms import ToTensor, Compose

from augmentations.augmentations import (
        GaussianDequantize,
        UniformDequantize,
        Reweight,
        GaussianFilter,
        MinMaxScaler,
)

logger = logging.getLogger("ptsemseg")


key2aug = {
        'GaussianDequantize': GaussianDequantize,
        'UniformDequantize': UniformDequantize,
        'totensor': ToTensor,
        'Reweight': Reweight,
        'GaussianFilter': GaussianFilter,
        'MinMaxScaler': MinMaxScaler,
        }


def get_composed_augmentations(aug_dict):
    if aug_dict is None:
        # print("Using No Augmentations")
        return None

    augmentations = []
    for aug_key, aug_param in aug_dict.items():
        augmentations.append(key2aug[aug_key](**aug_param))
        print("Using {} aug with params {}".format(aug_key, aug_param))
    return Compose(augmentations)
