"""Learnable outgoing signal generation.

The physical game action already has a deterministic observation encoding
(`build_signal`). This module models a separate communicative signal chosen by
the agent as part of an intervention.

The signal vector has no fixed labels. It gains meaning only through the
response model and downstream experience losses.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

import config


@dataclass(frozen=True)
class OutgoingSignal:
    """A learned communicative signal emitted with an action."""

    vector: torch.Tensor
    action_signal: torch.Tensor


class SignalGenerator(nn.Module):
    """Produces a communicative signal from belief and candidate action."""

    def __init__(self, input_dim=config.SIGNAL_INPUT_DIM,
                 signal_dim=config.SIGNAL_DIM):
        super().__init__()
        self.input_dim = input_dim
        self.signal_dim = signal_dim
        self.W_signal = nn.Linear(input_dim, signal_dim, bias=True)

    def forward(self, action_signal, responder_intent, actor_sigma):
        action_signal = torch.as_tensor(
            action_signal, dtype=responder_intent.dtype,
            device=responder_intent.device)
        actor_sigma = torch.as_tensor(
            actor_sigma, dtype=responder_intent.dtype,
            device=responder_intent.device)
        x = torch.cat([action_signal, responder_intent, actor_sigma], dim=-1)
        assert x.shape[-1] == self.input_dim, (x.shape, self.input_dim)
        return OutgoingSignal(
            vector=torch.tanh(self.W_signal(x)),
            action_signal=action_signal,
        )
