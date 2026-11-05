"""Isolate upstream projects which expose generic top-level module names."""
from contextlib import contextmanager
from pathlib import Path
import sys
from threading import RLock


_IMPORT_LOCK = RLock()


@contextmanager
def source_modules(path, roots):
    path = str(Path(path).resolve())
    if not Path(path).is_dir():
        raise ValueError(f"Model source directory does not exist: {path}")
    with _IMPORT_LOCK:
        saved = {key: value for key, value in sys.modules.items()
                 if key.split(".")[0] in roots}
        for key in saved:
            del sys.modules[key]
        sys.path.insert(0, path)
        try:
            yield
        finally:
            for key in list(sys.modules):
                if key.split(".")[0] in roots:
                    del sys.modules[key]
            sys.modules.update(saved)
            sys.path.remove(path)
