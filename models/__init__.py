import os
from omegaconf import OmegaConf
import copy
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from augmentations import get_composed_augmentations

from models.ae import AE
from models.nae import NAE
from models.modules import (
    DeConvNet2,
    FCNet,
    ConvNet2FC,
    ConvVAE,
    DeConvVAE,
    ConvNet2D,
    DeConvNet2D,
    ModConvVAE,
    ModDeConvVAE,
)


def get_net(in_dim, out_dim, **kwargs):
    nh = kwargs.get("nh", 8)
    out_activation = kwargs.get("out_activation", "linear")

    if kwargs["arch"] == "conv2fc":
        nh_mlp = kwargs["nh_mlp"]
        net = ConvNet2FC(
            in_chan=in_dim,
            out_chan=out_dim,
            nh=nh,
            nh_mlp=nh_mlp,
            out_activation=out_activation,
        )

    elif kwargs["arch"] == "deconv2":
        net = DeConvNet2(
            in_chan=in_dim, out_chan=out_dim, nh=nh, out_activation=out_activation
        )
    elif kwargs["arch"] == "convae":
        activation = kwargs["activation"]
        use_spectral_norm = kwargs["spectral_norm"]
        use_dropout = kwargs["dropout"]
        use_bnorm = kwargs["batch_norm"]
        bias = kwargs["bias"]
        net = ConvVAE(
            in_chan = in_dim,
            nh_bln = out_dim,
            nh = nh,
            out_activation = out_activation,
            activation = activation,
            use_spectral_norm = use_spectral_norm,
            use_dropout = use_dropout,
            use_bnorm = use_bnorm,
            bias = bias
        )
    elif kwargs["arch"] == "modconvae":
        activation = kwargs["activation"]
        use_spectral_norm = kwargs["spectral_norm"]
        use_bnorm = kwargs["batch_norm"]
        bias = kwargs["bias"]
        net = ModConvVAE(
            in_chan = in_dim,
            nh_bln = out_dim,
            nh = nh,
            out_activation = out_activation,
            activation = activation,
            bias = bias,
            use_bnorm = use_bnorm,
            use_spectral_norm = use_spectral_norm
        )
    elif kwargs["arch"] == "convae2":
        activation = kwargs["activation"]
        use_spectral_norm = kwargs["spectral_norm"]
        net = ConvNet2D(
            in_chan = in_dim,
            nh_bln = out_dim,
            nh = nh,
            out_activation = out_activation,
            activation = activation,
            use_spectral_norm = use_spectral_norm
        )
    elif kwargs["arch"] == "deconvae":
        activation = kwargs["activation"]
        use_spectral_norm = kwargs["spectral_norm"]
        use_dropout = kwargs["dropout"]
        bias = kwargs["bias"]
        net = DeConvVAE(
                nh_bln = in_dim,
                out_chan = out_dim,
                nh = nh,
                out_activation = out_activation,
                activation = activation,
                use_spectral_norm = use_spectral_norm,
                use_dropout = use_dropout,
                bias = bias
        )
    elif kwargs["arch"] == "moddeconvae":
        activation = kwargs["activation"]
        use_spectral_norm = kwargs["spectral_norm"]
        use_bnorm = kwargs["batch_norm"]
        bias = kwargs["bias"]
        net = ModDeConvVAE(
                nh_bln = in_dim,
                out_chan = out_dim,
                nh = nh,
                out_activation = out_activation,
                activation = activation,
                bias = bias,
                use_bnorm = use_bnorm,
                use_spectral_norm = use_spectral_norm
        )
    elif kwargs["arch"] == "deconvae2":
        activation = kwargs["activation"]
        use_spectral_norm = kwargs["spectral_norm"]
        net = DeConvNet2D(
                nh_bln = in_dim,
                out_chan = out_dim,
                nh = nh,
                out_activation = out_activation,
                activation = activation,
                use_spectral_norm = use_spectral_norm
        )
    elif kwargs["arch"] == "fc":
        l_hidden = kwargs["l_hidden"]
        activation = kwargs["activation"]
        enc_dec = kwargs["enc_dec"]
        use_dropout = kwargs["dropout"]
        net = FCNet(
            in_dim=in_dim,
            out_dim=out_dim,
            l_hidden=l_hidden,
            activation=activation,
            out_activation=out_activation,
            enc_dec=enc_dec,
            use_dropout=use_dropout
        )
    return net


def get_ae(**model_cfg):
    arch = model_cfg.pop('arch')
    x_dim = model_cfg.pop("x_dim")
    z_dim = model_cfg.pop("z_dim")
    enc_cfg = model_cfg.pop('encoder')
    dec_cfg = model_cfg.pop('decoder')

    if arch == "ae":
        encoder = get_net(in_dim=x_dim, out_dim=z_dim, **enc_cfg)
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **dec_cfg)
        ae = AE(encoder, decoder)
    return ae

def get_vae(**model_cfg):
    x_dim = model_cfg["x_dim"]
    z_dim = model_cfg["z_dim"]
    encoder_out_dim = z_dim * 2

    encoder = get_net(in_dim=x_dim, out_dim=encoder_out_dim, **model_cfg["encoder"])
    decoder = get_net(in_dim=z_dim, out_dim=x_dim, **model_cfg["decoder"])
    n_sample = model_cfg.get("n_sample", 1)
    pred_method = model_cfg.get("pred_method", "recon")

    if model_cfg["arch"] == "vae":
        ae = VAE(encoder, decoder, n_sample=n_sample, pred_method=pred_method)
    return ae

def get_nae(**model_cfg):
    arch = model_cfg.pop("arch")
    x_dim = model_cfg["x_dim"]
    z_dim = model_cfg["z_dim"]

    encoder = get_net(in_dim=x_dim, out_dim=z_dim, **model_cfg["encoder"])
    decoder = get_net(in_dim=z_dim, out_dim=x_dim, **model_cfg["decoder"])

    if arch == "nae":
        ae = NAE(encoder, decoder, **model_cfg["nae"])
    else:
        raise ValueError(f"{arch}")
    return ae


def get_model(cfg, *args, version=None, **kwargs):
    # cfg can be a whole config dictionary or a value of a key 'model' in the config dictionary (cfg['model']).
    if "model" in cfg:
        model_dict = cfg["model"]
    elif "arch" in cfg:
        model_dict = cfg
    else:
        raise ValueError(f"Invalid model configuration dictionary: {cfg}")
    name = model_dict["arch"]
    model = _get_model_instance(name)
    model = model(**model_dict)
    return model


def _get_model_instance(name):
    try:
        return {
            "ae": get_ae,
            "nae": get_nae,
        }[name]
    except:
        raise ("Model {} not available".format(name))


def load_pretrained(identifier, config_file, ckpt_file, root='pretrained', **kwargs):
    """
    load pre-trained model.
    identifier: '<model name>/<run name>'. e.g. 'ae_mnist/z16'
    config_file: name of a config file. e.g. 'ae.yml'
    ckpt_file: name of a model checkpoint file. e.g. 'model_best.pth'
    root: path to pretrained directory
    """
    config_path = os.path.join(root, identifier, config_file)
    ckpt_path = os.path.join(root, identifier, ckpt_file)
    cfg = OmegaConf.load(config_path)
    model_name = cfg['model']['arch']

    model = get_model(cfg)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'model_state' in ckpt:
        ckpt = ckpt['model_state']
    model.load_state_dict(ckpt)
    model.eval()
    return model, cfg

