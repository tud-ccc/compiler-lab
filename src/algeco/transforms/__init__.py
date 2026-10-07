"""Middle-end passes (xDSL).

Every pass is a `ModulePass` with a unique `name`. Register new passes in
`_algeco_passes` below so that `algeco -p <name>` and `algeco.jit(passes=...)`
can find them.
"""

from collections.abc import Callable

from xdsl.passes import ModulePass
from xdsl.transforms import get_all_passes as get_xdsl_passes


def _algeco_passes() -> dict[str, Callable[[], type[ModulePass]]]:
    def interchange():
        from algeco.transforms.interchange import InterchangePass

        return InterchangePass

    def tiling():
        from algeco.transforms.tiling import TileLoopsPass

        return TileLoopsPass

    def fusion():
        from algeco.transforms.fusion import FuseLoopsPass

        return FuseLoopsPass

    def vectorize():
        from algeco.transforms.vectorize import VectorizeLoopsPass

        return VectorizeLoopsPass

    return {
        "algeco-interchange": interchange,
        "algeco-tile": tiling,
        "algeco-fuse-loops": fusion,
        "algeco-vectorize": vectorize,
    }


def get_all_passes() -> dict[str, Callable[[], type[ModulePass]]]:
    """All passes known to algeco: upstream xDSL passes plus our own."""
    return {**get_xdsl_passes(), **_algeco_passes()}
