"""Built-in declarative game specifications."""

from specs.ultimatum import ULTIMATUM_SPEC, build_ultimatum_spec
from specs.prisoners_dilemma import PRISONERS_DILEMMA_SPEC
from specs.chicken import CHICKEN_SPEC
from specs.stag_hunt import STAG_HUNT_SPEC
from specs.public_goods import PUBLIC_GOODS_SPEC
from specs.first_price_auction import FIRST_PRICE_AUCTION_SPEC

BENCHMARK_SPECS = (
    ULTIMATUM_SPEC,
    PRISONERS_DILEMMA_SPEC,
    CHICKEN_SPEC,
    STAG_HUNT_SPEC,
    PUBLIC_GOODS_SPEC,
    FIRST_PRICE_AUCTION_SPEC,
)

__all__ = [
    "ULTIMATUM_SPEC", "build_ultimatum_spec",
    "PRISONERS_DILEMMA_SPEC", "CHICKEN_SPEC", "STAG_HUNT_SPEC",
    "PUBLIC_GOODS_SPEC", "FIRST_PRICE_AUCTION_SPEC", "BENCHMARK_SPECS",
]
