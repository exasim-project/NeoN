#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Smoke test for the installed NeoN Python package."""

from __future__ import annotations

import json
import sys
import traceback


def main() -> int:
    try:
        import neon
        import neon._neon as ext
    except Exception as exc:
        print("Failed to import NeoN package:", exc)
        traceback.print_exc()
        return 1

    info = {
        "version": getattr(neon, "__version__", "unknown"),
        "extension": getattr(ext, "__name__", "unknown"),
        "has_serial": getattr(neon, "__has_serial__", None),
        "has_cpu": getattr(neon, "__has_cpu__", None),
        "has_gpu": getattr(neon, "__has_gpu__", None),
    }
    print(json.dumps(info, indent=2))

    try:
        neon.initialize()
        neon.finalize()
    except Exception as exc:
        print("Runtime check failed:", exc)
        traceback.print_exc()
        return 2

    print("NeoN package smoke test: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
