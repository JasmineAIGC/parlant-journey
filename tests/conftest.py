"""
tests/conftest.py
~~~~~~~~~~~~~~~~~

Shared pytest configuration. Stubs out ``parlant.sdk`` so all test modules
can import ``journey.*`` without requiring the real Parlant package to be
installed.  The stub is injected before any test file is collected.
"""

from __future__ import annotations

import sys
import types


def _inject_parlant_stub() -> None:
    """Register a minimal ``parlant.sdk`` stub in ``sys.modules``."""
    if "parlant.sdk" in sys.modules:
        return  # already injected by a test file's own stub

    parlant = types.ModuleType("parlant")
    sdk = types.ModuleType("parlant.sdk")

    class CompositionMode:
        STRICT = "STRICT"
        FLUID = "FLUID"

    sdk.CompositionMode = CompositionMode
    sdk.Agent = object
    sdk.Journey = object
    sdk.Server = object

    parlant.sdk = sdk
    sys.modules.setdefault("parlant", parlant)
    sys.modules.setdefault("parlant.sdk", sdk)


_inject_parlant_stub()
