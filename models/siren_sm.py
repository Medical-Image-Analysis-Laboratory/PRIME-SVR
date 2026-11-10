# Siren of SirenSVR (MultiModalSVR/SirenSVR/models/siren.py, copied unchanged), used by the slice module
# (motion, slice scaling, slice weighting) exactly as in SirenSVR. The SR module uses models/siren.py.
import torch
import torch.nn as nn
import numpy as np


class Sine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


class SineLayer(nn.Module):
    def __init__(self, in_feat, out_feat, omega, first=False, linear=False):
        super().__init__()
        self.omega = omega if not linear else 1.0
        self.linear = nn.Linear(in_feat, out_feat)
        self.activation = Sine() if not linear else nn.Identity()
        with torch.no_grad():
            if first:
                self.linear.weight.uniform_(-1 / in_feat, 1 / in_feat)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / in_feat) / omega, np.sqrt(6 / in_feat) / omega)

    def forward(self, x):
        return self.activation(self.omega * self.linear(x))


class Siren(nn.Module):
    def __init__(self, in_dim, out_dim, n_units, n_hidden_layers, omega_0, last_linear, omega_schedule='increasing'):
        '''
        Args:
            in_dim: input dimension (e.g. 3 for 3D coordinates)
            out_dim: output dimension (e.g. 1 for intensity) (N for multi-channel output, e.g, multi-contrast)
            n_units: number of units in the hidden layers
            n_hidden_layers: number of hidden layers
            omega_0: initial omega
            last_linear: whether to use a linear layer at the end
            omega_schedule: schedule for the omegas (increasing, or vanilla)
        '''
        super().__init__()
        if omega_schedule == 'increasing':
            self.omegas = [omega_0 * (i+1) for i in range(n_hidden_layers+1)]
        elif omega_schedule == 'decreasing':
            self.omegas = [omega_0 * (n_hidden_layers+1-i) for i in range(n_hidden_layers+1)]
        else: # vanilla
            self.omegas = [omega_0] * (n_hidden_layers+1)

        self.net = nn.ModuleList()
        self.net.append(SineLayer(in_dim, n_units, self.omegas[0], first=True))
        for i in range(n_hidden_layers):
            self.net.append(SineLayer(n_units, n_units, self.omegas[i+1]))
        self.net.append(SineLayer(n_units, out_dim, self.omegas[-1], linear=last_linear))
    
    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x
