"""
smart_import.py  – stand-alone helper, no packages required
"""

from __future__ import annotations

import builtins
import importlib
import importlib.util
import inspect
import sys
from contextlib import ContextDecorator
from pathlib import Path
from types import ModuleType
from typing import Dict, Optional


# ─── internal helpers ──────────────────────────────────────────────────────── #

def _load(file_path: Path, mod_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def _resolve(module_path: str, caller_dir: Path) -> ModuleType:
    """
    Resolve *module_path*.

    If it begins with dots, interpret them relative to *caller_dir* **only**;
    no packages or parent walks are needed for the current test layout.
    """
    leading = len(module_path) - len(module_path.lstrip("."))
    remaining = module_path.lstrip(".")
    rel_parts = remaining.split(".") if remaining else []

    base = caller_dir
    for _ in range(max(leading - 1, 0)):
        base = base.parent

    if rel_parts:
        rel_path = base.joinpath(*rel_parts)
        pkg_init = rel_path / "__init__.py"
        mod_file = rel_path.with_suffix(".py")
    else:                                 # 'from . import x'
        pkg_init = base / "__init__.py"
        mod_file = base.with_suffix(".py")

    if pkg_init.is_file():
        hint = remaining or pkg_init.parent.name or "module"
        return _load(pkg_init, hint)
    if mod_file.is_file():
        # For relative imports, we need to register the module with the full name
        if module_path.startswith("."):
            # This is a relative import, register with the full name
            full_module_name = module_path.lstrip(".")
            return _load(mod_file, full_module_name)
        else:
            return _load(mod_file, mod_file.stem)

    # nothing on disk → last-chance absolute import
    return importlib.import_module(module_path.lstrip("."))


# ─── public API ────────────────────────────────────────────────────────────── #

def smart_import(
    stmt: str,
    ns: Optional[Dict[str, object]] = None,
    _caller_dir: Optional[Path] = None,   # private; guard passes this
) -> object:
    if ns is None:
        ns = inspect.currentframe().f_back.f_globals  # type: ignore[arg-type]

    caller_dir = _caller_dir or Path(inspect.stack()[1].filename).parent

    stmt = stmt.strip()

    if stmt.startswith("from "):                           # FROM … IMPORT …
        mod_path, names = stmt[5:].split(" import ", 1)
        mod = _resolve(mod_path, caller_dir)
        results: Dict[str, object] = {}
        for frag in (f.strip() for f in names.split(",")):
            orig, _, alias = frag.partition(" as ")
            alias = alias or orig
            obj = getattr(mod, orig)
            ns[alias] = obj
            results[alias] = obj
        return next(iter(results.values())) if len(results) == 1 else results

    if stmt.startswith("import "):                         # IMPORT …
        rest = stmt[7:]
        mod_path, *alias_part = rest.split(" as ")
        alias = alias_part[0].strip() if alias_part else mod_path.split(".")[-1]
        mod = _resolve(mod_path, caller_dir)
        ns[alias] = mod
        return mod

    raise ValueError(f"Unsupported import statement: {stmt!r}")


class smart_import_guard(ContextDecorator):
    """Retry *any* failing import in the `with` block via `smart_import`."""

    def __enter__(self):
        self._orig = builtins.__import__

        def patched(name, globals=None, locals=None, fromlist=(), level=0):
            try:
                return self._orig(name, globals, locals, fromlist, level)
            except ImportError as e:
                # Handle relative imports correctly
                if level > 0:
                    dots = "." * level
                    module_path = f"{dots}{name}"
                else:
                    module_path = name
                
                if fromlist:
                    stmt = f"from {module_path} import {', '.join(fromlist)}"
                else:
                    stmt = f"import {module_path}"
                
                g = globals or {}
                caller_dir = Path(g.get("__file__", "." )).parent.resolve()
                result = smart_import(stmt, g, _caller_dir=caller_dir)
                # For from imports, we need to return the module, not the imported objects
                if fromlist:
                    # Get the actual module name (without dots for relative imports)
                    actual_module_name = name if level == 0 else name
                    module = sys.modules.get(actual_module_name, result)
                    return module
                return result

        builtins.__import__ = patched  # type: ignore[assignment]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        builtins.__import__ = self._orig  # type: ignore[assignment]
        return False  # propagate exceptions
