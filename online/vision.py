"""Compatibility entry point for online.perception.vision."""
import importlib
import sys

module = importlib.import_module("online.perception.vision")
if __name__ == "__main__" and hasattr(module, "main"):
    module.main()
else:
    sys.modules[__name__] = module
