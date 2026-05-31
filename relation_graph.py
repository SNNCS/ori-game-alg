"""Structure ①  人际关系图  G = (V, E, R).

Spec (v5, section 一):
    G in R^(n x n x k),  G[i,j,:] = i's view of the (i -> j) relation.
    G[i,j,:] != G[j,i,:]  -- the asymmetry is the core resource of the game.

What the v5 doc did by hand:
    G[j,i,:] <- G[j,i,:] + alpha_slow * W_z^T (z* - z)        # alpha = 0.005

What we do instead (kept optimization):
    G is a single nn.Parameter. The (z* - z) -> edge gradient is produced by
    autograd through the interpretation engine, and one optimizer updates G
    together with every other weight. The clamp to [-G_CLIP, G_CLIP] is applied
    on read (not in place) so it never breaks the autograd graph.
"""

import torch
import torch.nn as nn

import config


class RelationGraph(nn.Module):
    def __init__(self, n_agents=config.N_AGENTS, k=config.K, init_std=0.1):
        super().__init__()
        self.n = n_agents
        self.k = k
        G = torch.randn(n_agents, n_agents, k) * init_std
        for i in range(n_agents):
            G[i, i, :] = 0.0                      # diagonal = self-view = 0
        self.G = nn.Parameter(G)
        self.register_buffer("g_clip", torch.tensor(float(config.G_CLIP)))

    def get_edge(self, i, j):
        """G[i,j,:] with a read-time clamp (autograd-safe, not in place)."""
        return torch.clamp(self.G[i, j, :], -self.g_clip, self.g_clip)

    def get_row(self, i):
        """G[i,:,:] -- i's relation matrix toward everyone, shape (n, k)."""
        return torch.clamp(self.G[i, :, :], -self.g_clip, self.g_clip)

    @torch.no_grad()
    def edge_variance(self):
        """Diagnostic: how differentiated the learned edges have become."""
        return float(self.G.var(dim=(0, 1)).mean().item())

    @torch.no_grad()
    def asymmetry(self, i, j):
        """||G[i,j,:] - G[j,i,:]|| -- the cognitive asymmetry between i and j."""
        return float(torch.linalg.norm(self.G[i, j, :] - self.G[j, i, :]).item())
