"""The compilation steps shared by the `algeco` driver and `algeco.jit`."""

from dataclasses import dataclass, field
from io import StringIO

from xdsl.context import Context
from xdsl.dialect_interfaces.op_asm import OpAsmDialectInterface
from xdsl.dialects import memref
from xdsl.dialects.builtin import (
    Attribute,
    Builtin,
    BytesAttr,
    DenseIntOrFPElementsAttr,
    DenseResourceAttr,
    ModuleOp,
)
from xdsl.parser import Parser
from xdsl.passes import PassPipeline
from xdsl.printer import Printer

from algeco.backend import (
    check_lowerable,
    lower_to_llvm_dialect,
    translate_to_llvm_ir,
)
from algeco.dump import Dump
from algeco.transforms import get_all_passes


def run_passes(
    ctx: Context, module: ModuleOp, passes: str, dump: Dump | None = None
) -> None:
    """Apply a comma-separated pass pipeline, e.g. "algeco-interchange,algeco-tile".

    With `dump`, the IR after each pass is written to the dump directory.
    """

    def dump_after(prev, module, _next):
        if prev is not None:
            dump.stage(prev.name, to_text(module))

    if passes:
        callback = dump_after if dump else None
        PassPipeline.parse_spec(get_all_passes(), passes, callback).apply(ctx, module)


# Globals with more bytes than this print their data at the end of the module.
INLINE_GLOBAL_BYTES = 64


@dataclass(eq=False, repr=False)
class ResourcePrinter(Printer):
    """Prints the data of large globals (e.g. weights) at the end of the module.

    The initial value of such a global is printed as a reference to a resource,
    and its bytes in the module's resource section, as hex::

        "memref.global"() <{sym_name = "cst_w1", ...,
                            initial_value = dense_resource<cst_w1> : tensor<...>}>
        ...
        {-#
          dialect_resources: { builtin: { cst_w1: "0x04000000..." } }
        #-}

    This is upstream MLIR syntax, so mlir-opt reads it. Only the printed text
    changes: in memory, the global keeps its `dense<...>` value. (xDSL's
    `memref.global` does not accept `dense_resource` values.)
    """

    # Initial values to print as resources: id of the attribute -> resource key.
    resource_keys: dict[int, str] = field(default_factory=dict)
    resources: dict[str, bytes] = field(default_factory=dict)

    def print_module(self, module: ModuleOp) -> None:
        for op in module.walk():
            value = getattr(op, "initial_value", None)
            if (
                isinstance(op, memref.GlobalOp)
                and isinstance(value, DenseIntOrFPElementsAttr)
                and len(value.data.data) > INLINE_GLOBAL_BYTES
            ):
                self.resource_keys[id(value)] = op.sym_name.data
        self.print_op(module)
        self.print_resources()

    def print_attribute(self, attribute: Attribute) -> None:
        key = self.resource_keys.get(id(attribute))
        if key is None:
            return super().print_attribute(attribute)
        assert isinstance(attribute, DenseIntOrFPElementsAttr)
        self.resources[key] = attribute.data.data
        self.print_string(f"dense_resource<{key}> : ")
        self.print_attribute(attribute.type)

    def print_resources(self) -> None:
        if not self.resources:
            return
        # A blob starts with its alignment, as a 4-byte little-endian integer.
        alignment = (4).to_bytes(4, "little")
        blobs = ",\n".join(
            f'      {key}: "0x{(alignment + data).hex().upper()}"'
            for key, data in self.resources.items()
        )
        self.print_string(
            "\n\n{-#\n  dialect_resources: {\n    builtin: {\n"
            f"{blobs}\n    }}\n  }}\n#-}}"
        )


def parse_module(ctx: Context, text: str, name: str = "<text>") -> ModuleOp:
    """Parse a module; globals printed as `dense_resource` get their data back."""
    # The parser keeps resource blobs in the builtin dialect, for the whole
    # process. A key parsed before would be renamed in the IR but not in the
    # storage, so start empty: the blobs are inlined right after parsing.
    Builtin.get_interface(OpAsmDialectInterface)._blob_storage.clear()
    module = Parser(ctx, text, name).parse_module()
    inline_resources(module)
    return module


def inline_resources(module: ModuleOp) -> None:
    """Give parsed globals with a `dense_resource` value their data back.

    The inverse of `ResourcePrinter`, for IR read from text such as a dump.
    The parser keeps the blobs in the builtin dialect.
    """
    blobs = Builtin.get_interface(OpAsmDialectInterface)
    for op in module.walk():
        value = getattr(op, "initial_value", None)
        if isinstance(op, memref.GlobalOp) and isinstance(value, DenseResourceAttr):
            key = value.resource_handle.data
            blob = blobs.build_resources([key]).get(key)
            if blob is None:
                raise ValueError(f"resource '{key}' has no data")
            data = bytes.fromhex(blob[len("0x") + 8 :])  # skip the alignment
            op.properties["initial_value"] = DenseIntOrFPElementsAttr(
                value.type, BytesAttr(data)
            )


def to_text(module: ModuleOp) -> str:
    buf = StringIO()
    ResourcePrinter(stream=buf).print_module(module)
    return buf.getvalue() + "\n"


def to_llvm_ir(module: ModuleOp, dump: Dump | None = None) -> str:
    """Hand a module on buffers to the backend (mlir-opt + mlir-translate)."""
    check_lowerable(module)
    llvm_dialect = lower_to_llvm_dialect(to_text(module))
    if dump:
        dump.stage("llvm-dialect", llvm_dialect)
    llvm_ir = translate_to_llvm_ir(llvm_dialect)
    if dump:
        dump.stage("llvm", llvm_ir, ".ll")
    return llvm_ir
