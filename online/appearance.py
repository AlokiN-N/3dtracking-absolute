"""Compatibility entry point for online.perception.appearance."""
import importlib
import sys

module = importlib.import_module("online.perception.appearance")
if __name__ == "__main__" and hasattr(module, "main"):
    module.main()
else:
    sys.modules[__name__] = module
