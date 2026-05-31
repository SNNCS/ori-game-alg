"""Situation block sigma_j = [ rho || h || omega || K ]  (dim = SIGMA_DIM).

These are the per-agent context pieces the interpretation engine consumes
alongside the signal, edge, and rule blocks:

    rho   role embedding              (RHO_DIM)
    h     EMA history summary         (H_DIM)
    omega resource state             (OMEGA_DIM)   omega[0] = physical payoff
    K     public-knowledge vector     (K_DIM)

Nothing here is e-commerce specific; the history one-hot uses the ultimatum
response triple (accept / reject / counter).
"""

import numpy as np
import torch
import torch.nn as nn

import config


class RoleEmbedding(nn.Module):
    def __init__(self, n_roles=config.N_ROLES, rho_dim=config.RHO_DIM):
        super().__init__()
        self.emb = nn.Embedding(n_roles, rho_dim)
        nn.init.normal_(self.emb.weight,
                        std=float(np.sqrt(2.0 / (n_roles + rho_dim))))

    def forward(self, role_id):
        idx = torch.tensor(int(role_id), device=self.emb.weight.device)
        return self.emb(idx)


class HistorySummarizer(nn.Module):
    """h_j <- gamma * h_j + (1 - gamma) * tanh(W_enc[s || resp_onehot || z_j]).

    resp_onehot is over (accept, reject, counter).
    """

    HIST_SIGNAL_DIM = config.M

    def __init__(self, dim=config.H_DIM, d=config.D):
        super().__init__()
        self.dim = dim
        self.d = d
        input_dim = self.HIST_SIGNAL_DIM + 3 + d
        self.W_enc = nn.Linear(input_dim, dim, bias=True)

    @staticmethod
    def response_onehot(response, device=None, dtype=torch.float32):
        idx = {"accept": 0, "reject": 1, "counter": 2}.get(response, 2)
        v = torch.zeros(3, device=device, dtype=dtype)
        v[idx] = 1.0
        return v

    def encode(self, signal_vec, response, z_j):
        signal_vec = torch.as_tensor(signal_vec, dtype=z_j.dtype, device=z_j.device)
        resp = self.response_onehot(response, device=z_j.device, dtype=z_j.dtype)
        x = torch.cat([signal_vec, resp, z_j], dim=-1)
        return torch.tanh(self.W_enc(x))

    def update_h(self, h_j, signal_vec, response, z_j, gamma=config.GAMMA):
        enc = self.encode(signal_vec, response, z_j)
        return gamma * h_j + (1.0 - gamma) * enc


# --- plain helpers ---------------------------------------------------------

def init_resource(role_id, initial_resource_map, dim=config.OMEGA_DIM, rng=None):
    """omega[0] carries the physical starting payoff; the rest is small noise."""
    rng = rng if rng is not None else np.random.default_rng()
    omega = torch.tensor(rng.normal(0, 0.01, dim), dtype=torch.float32)
    omega[0] = float(initial_resource_map.get(role_id, 0.0))
    return omega


def update_resource(omega_j, payoff_delta, clip=5.0):
    new_omega = omega_j.clone()
    new_omega[0] = new_omega[0] + float(payoff_delta)
    return torch.clamp(new_omega, -clip, clip)


def init_knowledge(dim=config.K_DIM, dtype=torch.float32):
    """Public-knowledge vector K. The mutual-knowledge term of the signal:
    everyone knows the pie is 1, the game is one-shot-with-counters, etc.
    """
    K = torch.zeros(dim, dtype=dtype)
    K[0] = 1.0     # total pie known to be 1
    K[1] = 1.0     # both players are rational observers
    K[2] = 1.0     # the responder may reject
    K[3] = 0.6     # observability of the offer
    if dim >= 8:
        K[4:8] = K[0:4] * 0.7
    return K


def build_sigma(rho, h, omega, K):
    return torch.cat([rho, h, omega, K], dim=-1)
