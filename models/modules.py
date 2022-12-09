import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.spectral_norm import spectral_norm

class DummyDistribution(nn.Module):
    """ Function-less class introduced for backward-compatibility of model checkpoint files. """
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer('sigma', torch.tensor(0., dtype=torch.float))

    def forward(self, x):
        return self.net(x)

class ConvNet2FC(nn.Module):
    """additional 1x1 conv layer at the top"""
    def __init__(self, in_chan=1, out_chan=64, nh=8, nh_mlp=512, out_activation='linear', use_spectral_norm=False):
        """nh: determines the numbers of conv filters"""
        super(ConvNet2FC, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, nh * 4, kernel_size=3, bias=True)
        self.conv2 = nn.Conv2d(nh * 4, nh * 8, kernel_size=3, bias=True)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(nh * 8, nh * 8, kernel_size=3, bias=True)
        self.conv4 = nn.Conv2d(nh * 8, nh * 16, kernel_size=3, bias=True)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(nh * 16, nh_mlp, kernel_size=7, bias=True)
        self.conv6 = nn.Conv2d(nh_mlp, out_chan, kernel_size=1, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan
        self.out_activation = get_activation(out_activation)

        if use_spectral_norm:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)
            self.conv3 = spectral_norm(self.conv3)
            self.conv4 = spectral_norm(self.conv4)
            self.conv5 = spectral_norm(self.conv5)

        layers = [self.conv1,
                  nn.ReLU(),
                  self.conv2,
                  nn.ReLU(),
                  self.max1,
                  self.conv3,
                  nn.ReLU(),
                  self.conv4,
                  nn.ReLU(),
                  self.max2,
                  self.conv5,
                  nn.ReLU(),
                  self.conv6,]
        if self.out_activation is not None:
            layers.append(self.out_activation)


        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DeConvNet2(nn.Module):
    def __init__(self, in_chan=1, out_chan=1, nh=8, out_activation='linear',
                 use_spectral_norm=False):
        """nh: determines the numbers of conv filters"""
        super(DeConvNet2, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_chan, nh * 16, kernel_size=7, bias=True)
        self.conv2 = nn.ConvTranspose2d(nh * 16, nh * 8, kernel_size=3, bias=True)
        self.conv3 = nn.ConvTranspose2d(nh * 8, nh * 8, kernel_size=3, bias=True)
        self.conv4 = nn.ConvTranspose2d(nh * 8, nh * 4, kernel_size=3, bias=True)
        self.conv5 = nn.ConvTranspose2d(nh * 4, out_chan, kernel_size=3, bias=True)
        self.in_chan, self.out_chan = in_chan, out_chan
        self.out_activation = get_activation(out_activation) 

        if use_spectral_norm:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)
            self.conv3 = spectral_norm(self.conv3)
            self.conv4 = spectral_norm(self.conv4)
            self.conv5 = spectral_norm(self.conv5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        return x

class SphericalActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / x.norm(p=2, dim=1, keepdim=True)


# Fully Connected Network
def get_activation(s_act):
    if s_act == 'relu':
        return nn.ReLU(inplace=True)
    elif s_act == 'PReLU':
        return nn.PReLU()
    elif s_act == 'sigmoid':
        return nn.Sigmoid()
    elif s_act == 'softplus':
        return nn.Softplus()
    elif s_act == 'linear':
        return None
    elif s_act == 'tanh':
        return nn.Tanh()
    elif s_act == 'leakyrelu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif s_act == 'softmax':
        return nn.Softmax(dim=1)
    elif s_act == 'swish':
        return nn.SiLU()
    elif s_act == 'spherical':
        return SphericalActivation()
    else:
        raise ValueError(f'Unexpected activation: {s_act}')


class FCNet(nn.Module):
    """fully-connected network"""
    def __init__(self, in_dim, out_dim, l_hidden=(50,), activation='sigmoid', out_activation='linear',
                 use_spectral_norm=False, enc_dec=1, use_dropout=False):
        super().__init__()
        l_neurons = tuple(l_hidden) + (out_dim,)
        if isinstance(activation, str):
            activation = (activation,) * len(l_hidden)
        activation = tuple(activation) + (out_activation,)

        l_layer = []
        prev_dim = in_dim
        if enc_dec:
            l_layer.append(nn.Flatten())

        for i_layer, (n_hidden, act) in enumerate(zip(l_neurons, activation)):
            if use_spectral_norm and i_layer < len(l_neurons) - 1:  # don't apply SN to the last layer
                l_layer.append(spectral_norm(nn.Linear(prev_dim, n_hidden)))
            else:
                l_layer.append(nn.Linear(prev_dim, n_hidden))
            act_fn = get_activation(act)
            if act_fn is not None:
                l_layer.append(act_fn)
            prev_dim = n_hidden
            l_layer.append(nn.Dropout(p=0.1))
        
        if not enc_dec:
            l_layer.append(View((-1, 1, 40, 40)))

        self.net = nn.Sequential(*l_layer)
        self.in_dim = in_dim
        self.out_shape = (out_dim,) 

    def forward(self, x):
        return self.net(x)

class ConvVAE(nn.Module):
    def __init__(self, in_chan=1, nh_bln=32, nh=5, out_activation='PReLU', activation='PReLU', use_spectral_norm=False, use_dropout=False, use_bnorm=False, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chan, nh*2, kernel_size=5, bias=bias, stride=1, padding=2)
        self.conv2 = nn.Conv2d(nh*2, nh*2, kernel_size=5, bias=bias, stride=2, padding=2)
        self.conv3 = nn.Conv2d(nh*2, nh, kernel_size=5, bias=bias, stride=1, padding=2)
        self.conv4 = nn.Conv2d(nh, nh*20, kernel_size=20, bias=bias, stride=1)
        self.bottleneck = nn.Conv2d(nh*20, nh_bln, kernel_size=1, bias=True, stride=1)
        
        if use_spectral_norm:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)
            self.conv3 = spectral_norm(self.conv3)
            self.conv4 = spectral_norm(self.conv4)
        
        #if use_bnorm:
        self.bnorm1 = nn.BatchNorm2d(nh*2)
        self.bnorm2 = nn.BatchNorm2d(nh*2)
        self.bnorm3 = nn.BatchNorm2d(nh*1)
 
        if use_dropout:
            self.drop1 = nn.Dropout(p=0.1)
            self.drop2 = nn.Dropout(p=0.1)
            self.drop3 = nn.Dropout(p=0.1)
            self.drop4 = nn.Dropout(p=0.1)

        layers = []

        layers.append(self.conv1)
        layers.append(get_activation(activation))
        if use_bnorm:
            layers.append(self.bnorm1)
        if use_dropout:
            layers.append(self.drop1)
        layers.append(self.conv2)
        layers.append(get_activation(activation))
        if use_bnorm:
            layers.append(self.bnorm2)
        if use_dropout:
            layers.append(self.drop2)
        layers.append(self.conv3)
        layers.append(get_activation(activation))
        if use_bnorm:
            layers.append(self.bnorm3)
        if use_dropout:
            layers.append(self.drop3)
        layers.append(self.conv4)
        layers.append(get_activation(activation))
        if use_dropout:
            layers.append(self.drop4)
        layers.append(self.bottleneck)

        if get_activation(out_activation) is not None:
            layers.append(get_activation(out_activation))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class DeConvVAE(nn.Module):
    def __init__(self, nh_bln=32, out_chan=1, nh=5, out_activation='PReLU', activation='PReLU', use_spectral_norm=False, use_dropout=False, use_bnorm=False, bias=True):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(nh_bln, nh*20, kernel_size=1, bias=bias, stride=1)
        self.deconv2 = nn.ConvTranspose2d(nh*20, nh, kernel_size=20, bias=bias, stride=1)
        self.deconv3 = nn.ConvTranspose2d(nh, nh*2, kernel_size=5, bias=bias, stride=2, padding=2, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(nh*2, nh*2, kernel_size=5, bias=bias, stride=1, padding=2)
        self.out = nn.ConvTranspose2d(nh*2, out_chan, kernel_size=5, bias=True, stride=1, padding=2)

        if use_spectral_norm:
            self.deconv1 = spectral_norm(self.deconv1)
            self.deconv2 = spectral_norm(self.deconv2)
            self.deconv3 = spectral_norm(self.deconv3)
            self.deconv4 = spectral_norm(self.deconv4)
        
        self.bnorm2 = nn.BatchNorm2d(nh*1)
        self.bnorm3 = nn.BatchNorm2d(nh*2)
        self.bnorm4 = nn.BatchNorm2d(nh*2)
 
        if use_dropout:
            self.drop1 = nn.Dropout(p=0.1)
            self.drop2 = nn.Dropout(p=0.1)
            self.drop3 = nn.Dropout(p=0.1)
            self.drop4 = nn.Dropout(p=0.1)

        layers = []

        layers.append(self.deconv1)
        layers.append(get_activation(activation))
        if use_dropout:
            layers.append(self.drop1)
        layers.append(self.deconv2)
        layers.append(get_activation(activation))
        if use_bnorm:
            layers.append(self.bnorm2)
        if use_dropout:
            layers.append(self.drop2)
        layers.append(self.deconv3)
        layers.append(get_activation(activation))
        if use_bnorm:
            layers.append(self.bnorm3)
        if use_dropout:
            layers.append(self.drop3)
        layers.append(self.deconv4)
        layers.append(get_activation(activation))
        if use_bnorm:
            layers.append(self.bnorm4)
        if use_dropout:
            layers.append(self.drop4)
        layers.append(self.out)
        
        if get_activation(out_activation) is not None:
            layers.append(get_activation(out_activation))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ConvNet2D(nn.Module):
    def __init__(self, in_chan=1, nh_bln=32, nh=5, d_nodes=100, activation='PReLU', out_activation='PReLU', use_spectral_norm=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chan, 2*nh, kernel_size=5, bias=True, stride=1, padding=2)
        self.conv2 = nn.Conv2d(2*nh, 2*nh, kernel_size=5, bias=True, stride=2, padding=2)
        self.conv3 = nn.Conv2d(2*nh, nh, kernel_size=5, bias=True, stride=1, padding=2)
        self.dense = nn.Linear(nh*20*20, d_nodes, bias=True)
        self.bottleneck = nn.Linear(d_nodes, nh_bln, bias=True)

        if use_spectral_norm:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)
            self.conv3 = spectral_norm(self.conv3)

        layers = []

        layers.append(self.conv1)
        layers.append(get_activation(activation))
        layers.append(self.conv2)
        layers.append(get_activation(activation))
        layers.append(self.conv3)
        layers.append(get_activation(activation))

        layers.append(nn.Flatten())
        layers.append(self.dense)
        layers.append(get_activation(activation))
        layers.append(self.bottleneck)
        layers.append(get_activation(out_activation))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
class DeConvNet2D(nn.Module):
    def __init__(self, nh_bln=32, out_chan=1, nh=5, d_nodes=100, activation='PReLU', out_activation='PReLU', use_spectral_norm=False):
        super().__init__()
        self.bottleneck = nn.Linear(nh_bln, d_nodes, bias=True)
        self.dense = nn.Linear(d_nodes, nh*20*20,  bias=True)
        self.deconv1 = nn.ConvTranspose2d(nh, nh, kernel_size=5, bias=True, stride=2, padding=2, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(nh, 2*nh, kernel_size=5, bias=True, stride=1, padding=2)
        self.deconv3 = nn.ConvTranspose2d(2*nh, out_chan, kernel_size=5, bias=True, stride=1, padding=2)

        if use_spectral_norm:
            self.deconv1 = spectral_norm(self.deconv1)
            self.deconv2 = spectral_norm(self.deconv2)
            self.deconv3 = spectral_norm(self.deconv3)


        layers = []

        layers.append(self.bottleneck)
        layers.append(get_activation(activation))
        layers.append(self.dense)
        layers.append(get_activation(activation))
        layers.append(View((-1, nh, 20, 20)))
        layers.append(self.deconv1)
        layers.append(get_activation(activation))
        layers.append(self.deconv2)
        layers.append(get_activation(activation))
        layers.append(self.deconv3)
        if get_activation(out_activation) is not None:
            layers.append(get_activation(out_activation))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class ModConvVAE(nn.Module):
    def __init__(self, in_chan=1, nh_bln=32, nh=5, out_activation='linear', activation='PReLU', bias=True, use_bnorm=False, use_spectral_norm=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chan, nh*2, kernel_size=5, bias=bias, stride=1,  padding=2)
        self.conv2 = nn.Conv2d(nh*2, nh, kernel_size=5, bias=bias, stride=1, padding=2)
        self.dense = nn.Linear(nh*20*20, 100, bias=True)
        self.bottleneck = nn.Linear(100, nh_bln, bias=True)
        
        self.avg1 = nn.MaxPool2d(2, stride=2)

        if use_spectral_norm:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)

        layers = []

        layers.append(self.conv1)
        layers.append(get_activation(activation))
        layers.append(self.avg1)
        layers.append(get_activation(activation))
        layers.append(self.conv2)
        layers.append(get_activation(activation))
        layers.append(nn.Flatten())
        layers.append(self.dense)
        layers.append(get_activation(activation))
        layers.append(self.bottleneck)

        if get_activation(out_activation) is not None:
            layers.append(get_activation(out_activation))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ModDeConvVAE(nn.Module):
    def __init__(self, nh_bln=32, out_chan=1, nh=5, out_activation='linear', activation='PReLU', bias=True, use_bnorm=False, use_spectral_norm=False):
        super().__init__()

        self.bottleneck = nn.Linear(nh_bln, 100, bias=True)
        self.dense = nn.Linear(100, nh*20*20, bias=bias)
        self.deconv1 = nn.ConvTranspose2d(nh, nh*2, kernel_size=5, bias=bias, stride=1, padding=2)
        self.deconv2 = nn.ConvTranspose2d(nh*2, out_chan, kernel_size=5, bias=bias, stride=1, padding=2)
        self.activation = get_activation(activation)
        self.out_activation = get_activation(out_activation)
        self.nh = nh

        if use_spectral_norm:
            self.deconv1 = spectral_norm(self.deconv1)
            self.deconv2 = spectral_norm(self.deconv2)

    def forward(self, x):
        x = self.bottleneck(x)
        x = self.activation(x)
        x = self.dense(x)
        x = self.activation(x)
        x = x.view(-1, self.nh, 20, 20)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.activation(x)
        x = self.deconv1(x)
        x = self.activation(x)
        x = self.deconv2(x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        return x


