"""Compatibility entry point for online.evaluation.evaluate."""
import importlib
import sys

module = importlib.import_module("online.evaluation.evaluate")
if __name__ == "__main__" and hasattr(module, "main"):
    module.main()
else:
    sys.modules[__name__] = module
