# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""The recorded v1 rows — the oracle the three pair-parity suites compare against.

``data/v1_rows/<suite>__<config>.npz`` holds, for one ``(operator, method)``
configuration, exactly what ``_v1_side(name)`` used to rebuild on the spot:

* the ``_BandContext`` arrays the numpy model of the functor reads —
  ``target``, ``normal``, ``sdf``, ``dx``, ``donor``, ``weight``, ``at_wall``,
  ``fluid`` and the four :class:`WallClosure` arrays;
* v1's ``BandRows`` themselves — ``stencil``, ``a``, ``nnz``, ``c``, and the
  three bookkeeping arrays — from which each suite's ``_v1_row`` canonicaliser
  reads a row;
* the per-arm neighbour test ``{(d, step): (index, is_fluid)}``;
* ``div``/``grad``'s face flux, and ``div``'s ``central`` flag.

**Why recorded and not rebuilt.** The producer — ``ibm/band.py``,
``ibm/band_rows.py`` and the 598 lines of numpy row assembly in
``schemes/boundary/ghost_cell.py`` — is deleted with the band
(``plans/IBM/design.md`` §1.3, §11). The *numbers* are not: they are the
acceptance bar the port was built to, and they are the one thing that must not
be re-derived from the code under test. Recording them freezes the oracle at
the last tree that could produce it, which is strictly stronger than keeping a
numpy re-implementation beside the kernel — a re-implementation drifts, a file
of bits cannot.

Every array is stored and returned **bit-exactly**: ``np.savez_compressed`` is
lossless, and nothing here converts a dtype. A row read back through
``rows.a[r]`` is the same ``float64`` v1 shipped, so ``_v1_row``'s bitwise
comparisons say exactly what they said before.

Recorded 2026-07-29, on the tree at NeoN ``91f3c39e75`` — the last one with
v1's rows in it. Regenerating them is not a maintenance operation: it would
need that tree back. If a fixture ever disagrees with the compiled pair, the
pair moved.
"""

import os

import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "data", "v1_rows")


class _Closure:
    """The four arrays of v1's :class:`WallClosure`, read-only."""

    __slots__ = ("value_linear", "value_constant", "grad_linear", "grad_constant")

    def __init__(self, z):
        self.value_linear = z["closure.value_linear"]
        self.value_constant = z["closure.value_constant"]
        self.grad_linear = z["closure.grad_linear"]
        self.grad_constant = z["closure.grad_constant"]


class _Ctx:
    """v1's ``_BandContext``, as recorded — the attributes the models read."""

    __slots__ = (
        "target",
        "normal",
        "sdf",
        "dx",
        "donor",
        "weight",
        "at_wall",
        "fluid",
        "closure",
    )

    def __init__(self, z):
        self.target = z["ctx.target"]
        self.normal = z["ctx.normal"]
        self.sdf = z["ctx.sdf"]
        self.dx = z["ctx.dx"]
        self.donor = z["ctx.donor"]
        self.weight = z["ctx.weight"]
        self.at_wall = z["ctx.at_wall"]
        self.fluid = z["ctx.fluid"]
        self.closure = _Closure(z)

    @property
    def nrows(self):
        return int(self.target.shape[0])


class _Rows:
    """v1's ``BandRows``, as recorded — what ``_v1_row`` slices."""

    __slots__ = ("target", "stencil", "a", "nnz", "c", "patch", "box_offset", "stride")

    def __init__(self, z):
        self.target = z["rows.target"]
        self.stencil = z["rows.stencil"]
        self.a = z["rows.a"]
        self.nnz = z["rows.nnz"]
        self.c = z["rows.c"]
        self.patch = z["rows.patch"]
        self.box_offset = z["rows.box_offset"]
        self.stride = int(z["rows.stride"])

    @property
    def nrows(self):
        return int(self.target.shape[0])


class _CoeffRows:
    """One recorded ``BandRows`` of the grad coeff-placement study."""

    __slots__ = ("target", "stencil", "a", "nnz", "c")

    def __init__(self, z, idx):
        self.target = z[f"coeff{idx}.target"]
        self.stencil = z[f"coeff{idx}.stencil"]
        self.a = z[f"coeff{idx}.a"]
        self.nnz = z[f"coeff{idx}.nnz"]
        self.c = z[f"coeff{idx}.c"]


def load_grad_coeff_rows(name):
    """``{coeff: rows}`` — v1's grad rows of one configuration at each recorded
    ``coeff``, for H-5's exposure measurement.

    v1 folds ``coeff`` into ``scale`` *before* every product; the frame
    multiplies the finished sum. The two agree bitwise iff ``coeff`` is a power
    of two, and the study that pins that exposure needs v1's rows at a
    non-power-of-two ``coeff`` — which only v1 could build, so they are recorded
    like the rest.
    """
    path = os.path.join(DATA, f"grad_coeff__{name}.npz")
    if not os.path.exists(path):
        raise AssertionError(f"no recorded grad coeff rows for {name} at {path}")
    with np.load(path) as z:
        coeffs = [float(c) for c in z["coeffs"]]
        return {c: _CoeffRows(z, idx) for idx, c in enumerate(coeffs)}


def load(suite, name):
    """``(ctx, rows, arms, extra)`` for one recorded configuration.

    ``extra`` is the suite's own tail: ``{}`` for ``laplacian``,
    ``{"flux": ...}`` for ``grad``, ``{"flux": ..., "central": ...}`` for
    ``div``.
    """
    path = os.path.join(DATA, f"{suite}__{name}.npz")
    if not os.path.exists(path):
        raise AssertionError(
            f"no recorded v1 rows for {suite}/{name} at {path}. The fixtures are the oracle "
            "and cannot be regenerated from this tree — v1's row builder was deleted with the "
            "band. A configuration added to CONFIGS needs its rows recorded on a tree that "
            "still has one."
        )
    with np.load(path) as z:
        ctx = _Ctx(z)
        rows = _Rows(z)
        arms = {}
        for d in range(3):
            for step, tag in ((1, "p"), (-1, "m")):
                arms[(d, step)] = (z[f"arms.{d}{tag}.index"], z[f"arms.{d}{tag}.fluid"])
        extra = {}
        if "flux" in z.files:
            extra["flux"] = z["flux"]
        if "central" in z.files:
            extra["central"] = bool(z["central"])
    return ctx, rows, arms, extra
