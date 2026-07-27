# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""The rows a boundary scheme emits, and their upload to the band kernel (B3).

One row per band cell, in the affine form the whole IBM design is built on
(``plans/IBM/row-contract.md``, v1)::

    out(target[r], n) = sum_{k < nnz[r]} a[r, k] * phi(stencil[r, k], n)
                        + constant_scale * c[r, n]

``a`` never absorbs the wall datum — **all** of it lives in ``c``, which is what
lets the implicit track ask for the linear part alone (``constant_scale = 0``)
and what makes a time-dependent wall value a ``c`` rebuild with ``a``
untouched.

:class:`BandRows` is pure numpy and is built by a boundary scheme;
:func:`band_table` copies it once into the C++ handle that the kernel reads::

    rows = scheme.rows(term, mesh.ibm, lev, ncomp, t)
    table = band_table(rows, mesh.ibm.grid_version)
    blockamr.apply_band_rows(out_mf, phi_mf, table, ncomp,
                             blockamr.BandMode.Overwrite, 1.0,
                             mesh.ibm.grid_version)

Rows are grouped **per local box in ``MFIterator`` order** and addressed by the
CSR-style ``box_offset`` — the convention :class:`~blockamr.ibm.band.Band`
already uses, so a band cell and its row are laid out the same way.
"""

from dataclasses import dataclass

import numpy as np

import blockamr


@dataclass(frozen=True)
class BandRows:
    """One boundary scheme's contribution on one level, as a flat row list.

    The arrays are validated on construction — shape, dtype, contiguity and the
    two structural invariants (``0 <= nnz <= stride``, ``box_offset`` a CSR
    partition of the rows) — because the kernel indexes them without bounds
    checks and a bad row is a wrong number rather than a crash.

    ``stencil`` entries with ``k >= nnz[r]`` are dead: never read, never
    bounds-checked, free to hold anything. A row with ``nnz = 0`` reads nothing
    and writes ``constant_scale * c`` alone — that is how a non-fluid cell is
    pinned and how a cell is zeroed.
    """

    target: np.ndarray  # int32 (n, 3), global index
    stencil: np.ndarray  # int32 (n, stride, 3), global index
    a: np.ndarray  # f64 (n, stride)
    nnz: np.ndarray  # int32 (n,)
    c: np.ndarray  # f64 (n, ncomp)
    patch: np.ndarray  # int32 (n,)
    box_offset: np.ndarray  # int32 (nbox + 1,), CSR, MFIterator order
    stride: int

    def __post_init__(self):
        stride = int(self.stride)
        if stride < 0:
            raise ValueError(f"BandRows: stride must be >= 0, got {stride}.")
        if self.target.ndim != 2 or self.target.shape[1] != 3:
            raise ValueError(
                f"BandRows: 'target' must have shape (nrows, 3), got {self.target.shape}."
            )
        n = int(self.target.shape[0])
        _check(self.target, "target", (n, 3), np.int32)
        _check(self.stencil, "stencil", (n, stride, 3), np.int32)
        _check(self.a, "a", (n, stride), np.float64)
        _check(self.nnz, "nnz", (n,), np.int32)
        _check(self.patch, "patch", (n,), np.int32)
        if self.c.ndim != 2 or self.c.shape[0] != n or self.c.shape[1] < 1:
            raise ValueError(
                f"BandRows: 'c' must have shape ({n}, ncomp) with ncomp >= 1, got {self.c.shape}."
            )
        _check(self.c, "c", self.c.shape, np.float64)
        _check_box_offset(self.box_offset, n)
        bad = np.argwhere((self.nnz < 0) | (self.nnz > stride))
        if bad.size:
            row = int(bad[0, 0])
            raise ValueError(
                f"BandRows: 'nnz' must satisfy 0 <= nnz <= stride = {stride}, "
                f"got {int(self.nnz[row])} at row {row}."
            )

    @property
    def nrows(self):
        """Number of rows on this level."""
        return int(self.target.shape[0])

    @property
    def ncomp(self):
        """Components the affine constant carries."""
        return int(self.c.shape[1])

    @property
    def nbox(self):
        """Number of local boxes the rows are grouped into."""
        return int(self.box_offset.shape[0]) - 1


def band_table(rows, grid_version):
    """Copy ``rows`` into the C++ handle the band kernel reads.

    ``grid_version`` is the **IBM** generation (``mesh.ibm.grid_version``), the
    one bumped by a regrid *and* by re-assigning ``mesh.bodies`` — not
    ``mesh.grid_version``. Whatever is passed here is what
    ``apply_band_rows`` must be passed too: a mismatch is how a table that
    outlived its geometry raises instead of computing plausible wrong numbers.

    The upload is not free (one host-to-device copy per array), so the result
    belongs in a cache keyed by the row identity, not in the inner loop.
    """
    return blockamr.BandTable(
        rows.target,
        rows.stencil,
        rows.a,
        rows.nnz,
        rows.c,
        rows.patch,
        rows.box_offset,
        grid_version,
    )


def pin_rows(grids, geometries, ncomp):
    """The rows that pin every non-fluid cell to ``non_fluid_pin`` (B7).

    The interior sweep runs over the whole valid box, so at a band cell it
    reads its non-fluid neighbours. The result there is overwritten by the
    boundary scheme's row, but the read itself must not hit a trap value — so
    preprocessing writes the pin into those cells once, before the first sweep
    of an evaluate (design §7).

    Every row is ``nnz = 0``: it reads nothing and writes ``c`` alone. That is
    what lets the *same* MultiFab be passed as source and destination
    (row-contract §7), which is how a device-resident field is pinned without a
    second kernel — and it makes the pin idempotent by construction.

    ``grids`` and ``geometries`` are one per local box, in ``MFIterator``
    order.
    """
    cells, constants, patches, counts = [], [], [], []
    for grid, geometry in zip(grids, geometries):
        selected = geometry.depth <= 0
        cells.append(np.argwhere(selected) + np.asarray(grid.lo))
        counts.append(int(selected.sum()))
        constants.append(np.full((counts[-1], ncomp), float(geometry.non_fluid_pin)))
        patches.append(geometry.patch[selected])
    target = _rows_concat(cells, (0, 3), np.int32)
    nrows = target.shape[0]
    return BandRows(
        target=target,
        # stride 1 with a dead slot: the slot is never dereferenced (nnz = 0),
        # and pointing it at the row's own target keeps it inside every bound
        # the handle checks.
        stencil=np.ascontiguousarray(target[:, np.newaxis, :]),
        a=np.zeros((nrows, 1)),
        nnz=np.zeros(nrows, dtype=np.int32),
        c=_rows_concat(constants, (0, ncomp), np.float64),
        patch=_rows_concat(patches, (0,), np.int32),
        box_offset=np.concatenate([[0], np.cumsum(counts)]).astype(np.int32),
        stride=1,
    )


def _rows_concat(blocks, empty_shape, dtype):
    if not blocks:
        return np.zeros(empty_shape, dtype=dtype)
    return np.ascontiguousarray(np.concatenate(blocks), dtype=dtype)


def _check(array, name, shape, dtype):
    if array.dtype != dtype:
        raise ValueError(
            f"BandRows: '{name}' must have dtype {np.dtype(dtype).name}, got {array.dtype}."
        )
    if array.shape != tuple(shape):
        raise ValueError(f"BandRows: '{name}' must have shape {tuple(shape)}, got {array.shape}.")
    if not array.flags["C_CONTIGUOUS"]:
        raise ValueError(f"BandRows: '{name}' must be C-contiguous (use numpy.ascontiguousarray).")


def _check_box_offset(box_offset, nrows):
    """CSR: starts at 0, non-decreasing, ends at the row count."""
    if box_offset.ndim != 1 or box_offset.shape[0] < 1:
        raise ValueError(
            f"BandRows: 'box_offset' must have shape (nbox + 1,) with nbox >= 0, "
            f"got {box_offset.shape}."
        )
    _check(box_offset, "box_offset", box_offset.shape, np.int32)
    if box_offset[0] != 0:
        raise ValueError(f"BandRows: 'box_offset' must start at 0, got {int(box_offset[0])}.")
    if np.any(np.diff(box_offset) < 0):
        step = int(np.argwhere(np.diff(box_offset) < 0)[0, 0])
        raise ValueError(
            f"BandRows: 'box_offset' must be non-decreasing, got {int(box_offset[step + 1])} "
            f"after {int(box_offset[step])} at box {step}."
        )
    if int(box_offset[-1]) != nrows:
        raise ValueError(
            f"BandRows: 'box_offset' must end at nrows = {nrows}, got {int(box_offset[-1])}."
        )
