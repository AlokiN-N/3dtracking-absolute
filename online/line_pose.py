"""Compatibility entry point for online.estimation.line_pose."""
import importlib
import sys

module = importlib.import_module("online.estimation.line_pose")
if __name__ == "__main__" and hasattr(module, "main"):
    module.main()
else:
    sys.modules[__name__] = module
