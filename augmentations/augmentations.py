import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d


class UniformDequantize:
    def __init__(self):
        pass

    def __call__(self, img):
        img = img / 256. * 255. + torch.rand_like(img) / 256.
        return img

class Reweight:
    def __init__(self, n=1):
        self.n = n
        pass

    def __call__(self, img):
        img = img**self.n
        return img/img.sum()

class GaussianFilter:
    def __init__(self, sigma):
        self.sigma = sigma
        pass

    def __call__(self, img):
        for i in (1,2):
            img = gaussian_filter1d(img, self.sigma, axis=i)
        return img

class GaussianDequantize:
    def __init__(self):
        pass

    def __call__(self, img):
        img = img + torch.randn_like(img) * 0.01
        return img

class MinMaxScaler:
    def __init__(self):
        pass

    def __call__(self, img):
        img = img.view(1, -1)
        img -= img.min(1, keepdim=True)[0]
        img /= img.max(1, keepdim=True)[0]
        img = img.view(1, 40, 40)
        return img
