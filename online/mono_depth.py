"""Compatibility entry point for online.estimation.mono_depth."""
import importlib
import sys

module = importlib.import_module("online.estimation.mono_depth")
if __name__ == "__main__" and hasattr(module, "main"):
    module.main()
else:
    sys.modules[__name__] = module
