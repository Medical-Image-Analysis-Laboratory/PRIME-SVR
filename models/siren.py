import torch
import torch.nn as nn
import numpy as np
from utils.transform_utils import *


class SineLayer(nn.Module):
    # adapted and modified from Sitzmann et al. 2020, Implicit Neural Representations
    # with Periodic Activation Functions
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        intermed = self.linear(input)
        return torch.sin(self.omega_0 * intermed)


class ReluLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
    def forward(self, input):
        intermed = self.linear(input)
        return torch.relu(intermed)


class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class Siren(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, num_layers, f_om, h_om, outermost_linear, activation='sine'):
        super().__init__()
        layer = SineLayer if activation == 'sine' else ReluLayer
        self.net = nn.ModuleList()
        self.net.append(layer(in_size, hidden_size, is_first=True, omega_0=f_om))
        self.hidden_size = hidden_size
        for i in range(num_layers):
            self.net.append(layer(hidden_size, hidden_size, is_first=False, omega_0=h_om))

        if outermost_linear:
            final_linear = nn.Linear(hidden_size, out_size)
            if activation == 'sine':
                with torch.no_grad():
                    final_linear.weight.uniform_(-np.sqrt(6 / hidden_size) / h_om,
                                        np.sqrt(6 / hidden_size) / h_om)
            self.net.append(final_linear)
        else:
            self.net.append(layer(hidden_size, out_size, is_first=False, omega_0=h_om))

    def forward(self, coords):
        for ly in self.net[:-1]:
            coords = ly(coords)
        output = self.net[-1](coords)
        return output


class MLP(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, num_layers):
        super().__init__()

        self.net = [nn.Linear(in_size, hidden_size, bias=True), nn.ReLU()]
        self.hidden_size = hidden_size
        for i in range(num_layers):
            self.net.append(nn.Linear(hidden_size, hidden_size, bias=True))
            self.net.append(nn.ReLU())

        self.net.append(nn.Linear(hidden_size, out_size, bias=True))
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output
    


class MultiSiren(nn.Module):
    """
    3 parallel, non-connected SIRENs.
    Each outputs a scalar. Regularization is computed outside (or inside via helper).
    """
    def __init__(self, in_size, out_size, hidden_size, num_layers, f_om, h_om,
                 outermost_linear=True, activation='sine'):
        super().__init__()
        self.n_heads = out_size
        self.heads = nn.ModuleList([
            Siren(
                in_size=in_size,
                out_size=1,                 # scalar output per network
                hidden_size=hidden_size,
                num_layers=num_layers,
                f_om=f_om,
                h_om=h_om,
                outermost_linear=outermost_linear,
                activation=activation
            )
            for _ in range(out_size)
        ])

    def forward(self, coords):
        # Each head returns [B, 1] (or [*, 1]); concat -> [B, 3]
        outs = [h(coords) for h in self.heads]
        return torch.cat(outs, dim=-1)  # shape (..., out_size)

