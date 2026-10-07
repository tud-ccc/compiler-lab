"""Construction of the xDSL context used throughout algeco."""

from xdsl.context import Context
from xdsl.dialects import get_all_dialects


def make_context() -> Context:
    """Return a context in which all upstream xDSL dialects are loadable.

    If you define your own dialects, register them here as well.
    """
    ctx = Context()
    for name, factory in get_all_dialects().items():
        ctx.register_dialect(name, factory)
    return ctx
