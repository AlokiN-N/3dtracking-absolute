"""Compatibility entry point for online.evaluation.compare."""
import importlib
import sys

module = importlib.import_module("online.evaluation.compare")
if __name__ == "__main__" and hasattr(module, "main"):
    module.main()
else:
    sys.modules[__name__] = module
