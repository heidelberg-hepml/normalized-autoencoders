"""
ae.py
=====
Autoencoders
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision.utils import make_grid

class AE(nn.Module):
    """autoencoder"""
    def __init__(self, encoder, decoder):
        """
        encoder, decoder : neural networks
        """
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.own_optimizer = False

    def forward(self, x):
        z = self.encode(x)
        recon = self.decoder(z)
        return recon

    def encode(self, x):
        z = self.encoder(x)
        return z

    def predict(self, x):
        """one-class anomaly prediction"""
        recon = self(x)
        if hasattr(self.decoder, 'error'):
            predict = self.decoder.error(x, recon)
        else:
            predict = ((recon - x) ** 2).view(len(x), -1).mean(dim=1)
        return predict

    def predict_and_reconstruct(self, x):
        recon = self(x)
        if hasattr(self.decoder, 'error'):
            recon_err = self.decoder.error(x, recon)
        else:
            recon_err = ((recon - x) ** 2).view(len(x), -1).mean(dim=1)
        return recon_err, recon

    def validation_step(self, x, **kwargs):
        recon = self(x)
        if hasattr(self.decoder, 'error'):
            predict = self.decoder.error(x, recon)
        else:
            predict = ((recon - x) ** 2).view(len(x), -1).mean(dim=1)
        loss = predict.mean()

        if kwargs.get('show_image', True):
            x_img = make_grid(x.detach().cpu(), nrow=10, range=(0, 1))
            recon_img = make_grid(recon.detach().cpu(), nrow=10, range=(0, 1))
        else:
            x_img, recon_img = None, None
        return {'loss': loss.item(), 'predict': predict, 'reconstruction': recon,
                'input@': x_img, 'recon@': recon_img}

    def train_step(self, x, optimizer, clip_grad=None, **kwargs):
        optimizer.zero_grad()
        recon_error = self.predict(x)
        loss = recon_error.mean()
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_grad)
        optimizer.step()
        return {'loss': loss.item()}

    def reconstruct(self, x):
        return self(x)

    def sample(self, N, z_shape=None, device='cpu'):
        if z_shape is None:
            z_shape = self.encoder.out_shape

        rand_z = torch.rand(N, *z_shape).to(device) * 2 - 1
        sample_x = self.decoder(rand_z)
        return sample_x



def clip_vector_norm(x, max_norm):
    norm = x.norm(dim=-1, keepdim=True)
    x = x * ((norm < max_norm).to(torch.float) + (norm > max_norm).to(torch.float) * max_norm/norm + 1e-6)
    return x


