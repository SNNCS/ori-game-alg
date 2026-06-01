"""Compatibility adapter for the current Ultimatum Game.

Concrete game mechanics now live in GameSpec objects interpreted by
GenericGameAdapter. This module keeps the old UltimatumGameAdapter import path
stable while delegating to the generic implementation.
"""

from __future__ import annotations

import config
from game_spec import ActionAffordance, EntitySet, GroundedAction
from generic_adapter import GenericGameAdapter
from specs.ultimatum import build_ultimatum_spec


class UltimatumGameAdapter(GenericGameAdapter):
    """Backward-compatible wrapper around the declarative Ultimatum spec."""

    def __init__(self, rule=None, bids=config.BIDS, responses=config.RESPONSES):
        outside = (
            getattr(rule, "outside", config.OUTSIDE_OPTION)
            if rule is not None else config.OUTSIDE_OPTION)
        spec = build_ultimatum_spec(
            bids=bids,
            responses=responses,
            outside=outside,
            counter_discount=config.COUNTER_DISCOUNT,
            paths_open_decay=config.PATHS_OPEN_DECAY,
        )
        super().__init__(spec)
        if rule is not None:
            self.rule.r_public = rule.r_public
            self.rule.outside = outside


__all__ = [
    "ActionAffordance",
    "EntitySet",
    "GroundedAction",
    "UltimatumGameAdapter",
]
