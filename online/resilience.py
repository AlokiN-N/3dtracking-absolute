"""Compatibility entry point for online.perception.resilience."""
import importlib
import sys

module = importlib.import_module("online.perception.resilience")
if __name__ == "__main__" and hasattr(module, "main"):
    module.main()
else:
    sys.modules[__name__] = module
