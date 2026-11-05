"""Compatibility entry point for online.evaluation.plot."""
import importlib
import sys

module = importlib.import_module("online.evaluation.plot")
if __name__ == "__main__" and hasattr(module, "main"):
    module.main()
else:
    sys.modules[__name__] = module
