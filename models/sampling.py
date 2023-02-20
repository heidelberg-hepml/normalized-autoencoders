import numpy as np
import torch
import torch.autograd as autograd
import random

def clip_vector_norm(x, max_norm):
    norm = x.norm(dim=-1, keepdim=True)
    x = x * ((norm < max_norm).to(torch.float) + (norm > max_norm).to(torch.float) * max_norm/norm + 1e-6)
    return x

class SampleBufferV2:
    def __init__(self, max_samples=10000, replay_ratio=0.95):
        self.max_samples = max_samples
        self.buffer = []
        self.replay_ratio = replay_ratio

    def __len__(self):
        return len(self.buffer)

    def push(self, samples):
        samples = samples.detach().to('cpu')

        for sample in samples:
            self.buffer.append(sample)

            if len(self.buffer) > self.max_samples:
                self.buffer.pop(0)

    def get(self, n_samples):
        samples = random.choices(self.buffer, k=n_samples)
        samples = torch.stack(samples, 0)
        return samples

def sample_langevin_v2(x, model, stepsize, n_steps, noise_scale=None, intermediate_samples=False,
                    clip_x=None, clip_grad=None, reject_boundary=False, noise_anneal=None, noise_anneal_full=None,
                    spherical=False, mh=False, temperature=None, norm=False):
    """Langevin Monte Carlo
    x: torch.Tensor, initial points
    model: An energy-based model. returns energy
    stepsize: float
    n_steps: integer
    noise_scale: Optional. float. If None, set to np.sqrt(stepsize * 2)
    clip_x : tuple (start, end) or None boundary of square domain
    reject_boundary: Reject out-of-domain samples if True. otherwise clip.
    """
    assert not ((stepsize is None) and (noise_scale is None)), 'stepsize and noise_scale cannot be None at the same time'
    if noise_scale is None:
        noise_scale = np.sqrt(stepsize * 2)
    if stepsize is None:
        stepsize = (noise_scale ** 2) / 2
    noise_scale_ = noise_scale
    stepsize_ = stepsize
    if temperature is None:
        temperature = 1.

    # initial data
    x.requires_grad = True
    E_x = model(x)
    grad_E_x = autograd.grad(E_x.sum(), x, only_inputs=True)[0]
    if clip_grad is not None:
        grad_E_x = clip_vector_norm(grad_E_x, max_norm=clip_grad)
    E_y = E_x; grad_E_y = grad_E_x;

    l_samples = [x.detach().to('cpu')]
    l_dynamics = []; l_drift = []; l_diffusion = []; l_accept = []
    for i_step in range(n_steps):
        noise = torch.randn_like(x) * noise_scale_
        dynamics = - stepsize_ * grad_E_x / temperature + noise
        y = x + dynamics
        reject = torch.zeros(len(y), dtype=torch.bool)

        if clip_x is not None:
            if reject_boundary:
                accept = ((y >= clip_x[0]) & (y <= clip_x[1])).view(len(x), -1).all(dim=1)
                reject = ~ accept
                y[reject] = x[reject]
            else:
                y = torch.clamp(y, clip_x[0], clip_x[1])
        
        if norm:
            y = y/y.sum(dim=(2,3)).view(-1,1,1,1)
            
        if spherical:
            y = y / y.norm(dim=1, p=2, keepdim=True)

        # y_accept = y[~reject]
        # E_y[~reject] = model(y_accept)
        # grad_E_y[~reject] = autograd.grad(E_y.sum(), y_accept, only_inputs=True)[0]
        E_y = model(y)
        grad_E_y = autograd.grad(E_y.sum(), y, only_inputs=True)[0]
 
        if clip_grad is not None:
            grad_E_y = clip_vector_norm(grad_E_y, max_norm=clip_grad)

        if mh:
            y_to_x = ((grad_E_x + grad_E_y) * stepsize_ - noise).view(len(x), -1).norm(p=2, dim=1, keepdim=True) ** 2
            x_to_y = (noise).view(len(x), -1).norm(dim=1, keepdim=True, p=2) ** 2
            transition = - (y_to_x - x_to_y) / 4 / stepsize_  # B x 1
            prob = -E_y + E_x
            accept_prob = torch.exp((transition + prob) / temperature)[:,0]  # B
            reject = (torch.rand_like(accept_prob) > accept_prob) # | reject
            y[reject] = x[reject]
            E_y[reject] = E_x[reject]
            grad_E_y[reject] = grad_E_x[reject]
            x = y; E_x = E_y; grad_E_x = grad_E_y
            l_accept.append(~reject)

        x = y; E_x = E_y; grad_E_x = grad_E_y

        if noise_anneal is not None:
            noise_scale_ = noise_scale / (1 + i_step)

        elif noise_anneal_full is not None:
            noise_scale_ = noise_scale / np.sqrt(1 + i_step)
            stepsize_ = stepsize / (1 + i_step)

        l_dynamics.append(dynamics.detach().cpu())
        l_drift.append((- stepsize * grad_E_x).detach().cpu())
        l_diffusion.append(noise.detach().cpu())
        l_samples.append(x.detach().cpu())
    
    return {'sample': x.detach(), 'l_samples': l_samples, 'l_dynamics': l_dynamics,
            'l_drift': l_drift, 'l_diffusion': l_diffusion, 'l_accept': l_accept}

