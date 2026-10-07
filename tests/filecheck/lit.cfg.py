# ruff: noqa: F821  (`config` is injected by lit)
import lit.formats

config.name = "algeco"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir", ".py"]
config.excludes = ["lit.cfg.py"]
