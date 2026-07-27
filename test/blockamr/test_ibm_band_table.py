# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""The band row format and its kernel — ``BandRows``, ``BandTable``, ``apply_band_rows``.

This is the seam the whole IBM design rests on: every boundary scheme's output
is a list of affine rows, and this one kernel applies them::

    out(target[r], n) = sum_{k < nnz[r]} a[r, k] * phi(stencil[r, k], n)
                        + constant_scale * c[r, n]

Every test here is written against ``plans/IBM/row-contract.md`` and names the
frozen behaviour it pins, because everything downstream is Python-only and
cannot add to this surface without another compile.

**Why this file is unit-level while the rest of the IBM suite is not.** Same
reason as ``test_ibm_mesh.py``: the equation-only rule of
``plans/IBM/verification.md`` §1 governs the transferred *equation* suite,
which asserts physics through ``evaluate``. A row is not physics — it is an
index, a coefficient and a constant — and routing it through a laplacian would
test the laplacian.

**The numbers are exact.** ``phi`` is seeded with ``100*i + 10*j + k + 1000*n``
and every coefficient is a dyadic rational, so each expectation below is an
integer or a multiple of 1/8: representable in binary64, computed by hand in
the comment above the assert, and compared with ``==`` rather than a tolerance.

**Dead stencil slots hold poison.** Slots at ``k >= nnz`` are filled with an
index far outside any fab and a coefficient of 1e30, so a kernel that read one
would either segfault or return an absurd number. That the results below are
exact is the proof that it never does.

**No skip guard.** If the compiled extension predates ``BandTable``, these
tests fail loudly rather than skipping — a silently absent kernel is exactly
the failure this file exists to prevent.
"""

import numpy as np
import pytest

import blockamr
from blockamr.ibm.band_rows import BandRows, band_table

#: Cells per side of the test mesh.
N = 8

#: An index no fab reaches, and a coefficient no result could hide — what a
#: dead stencil slot holds.
POISON_INDEX = 10_000
POISON_COEFF = 1e30

#: The IBM generation every table here is built for; the kernel is handed the
#: same number unless the test is about staleness.
VERSION = 3


def _phi_value(i, j, k, n):
    """The seeded field: digit-separated so a wrong index is unmistakable."""
    return 100.0 * i + 10.0 * j + k + 1000.0 * n


def _full_box():
    return blockamr.Box([0, 0, 0], [N - 1, N - 1, N - 1])


def _split_boxes():
    """The 8^3 domain cut in two along x — the two-local-box layout."""
    return [
        blockamr.Box([0, 0, 0], [3, N - 1, N - 1]),
        blockamr.Box([4, 0, 0], [N - 1, N - 1, N - 1]),
    ]


def _box_array(boxes):
    box_list = blockamr.BoxList()
    for box in boxes:
        box_list.push_back(box)
    return blockamr.BoxArray(box_list)


def _multifab(boxes=None, ncomp=1, ngrow=1, memory="default"):
    """A zero-filled MultiFab over ``boxes`` (the whole domain by default)."""
    ba = _box_array(boxes if boxes is not None else [_full_box()])
    mf = blockamr.MultiFab(ba, blockamr.DistributionMapping(ba), ncomp, ngrow, memory=memory)
    mf.set_val(0.0)
    return mf


def _seed(mf):
    """Write ``_phi_value`` into every valid cell of ``mf``."""
    for mfi in blockamr.MFIterator(mf):
        lo = mfi.valid_box().small_end()
        arr = mf.copy_to_host(mfi)
        ni, nj, nk, ncomp = arr.shape
        i = np.arange(ni)[:, None, None] + lo[0]
        j = np.arange(nj)[None, :, None] + lo[1]
        k = np.arange(nk)[None, None, :] + lo[2]
        for n in range(ncomp):
            arr[:, :, :, n] = _phi_value(i, j, k, n)
        mf.copy_from(mfi, arr)
    return mf


def _values(mf):
    """The valid-region data of every local box, in ``MFIterator`` order."""
    return [mf.copy_to_host(mfi) for mfi in blockamr.MFIterator(mf)]


def _band_rows(rows, stride, box_offset=None):
    """Pack hand-written rows into :class:`BandRows`.

    ``rows`` is a list of ``(target, [(cell, coefficient), ...], c)``; the
    stencil slots past the live ones get poison, and the patch id is the row
    index unless a test cares about it.
    """
    nrows = len(rows)
    ncomp = len(rows[0][2]) if rows else 1
    target = np.zeros((nrows, 3), dtype=np.int32)
    stencil = np.full((nrows, stride, 3), POISON_INDEX, dtype=np.int32)
    coeff = np.full((nrows, stride), POISON_COEFF, dtype=np.float64)
    nnz = np.zeros(nrows, dtype=np.int32)
    constant = np.zeros((nrows, ncomp), dtype=np.float64)
    for r, (cell, live, c) in enumerate(rows):
        target[r] = cell
        nnz[r] = len(live)
        constant[r] = c
        for k, (donor, weight) in enumerate(live):
            stencil[r, k] = donor
            coeff[r, k] = weight
    if box_offset is None:
        box_offset = [0, nrows]
    return BandRows(
        target=target,
        stencil=stencil,
        a=coeff,
        nnz=nnz,
        c=constant,
        patch=np.arange(nrows, dtype=np.int32),
        box_offset=np.asarray(box_offset, dtype=np.int32),
        stride=stride,
    )


#: Three rows on the single-box mesh, exercising a two-term row, an empty row
#: and a three-term row. Their hand-computed values, at constant_scale = 1:
#:   (1, 1, 1): 2*phi(2,1,1) - phi(1,2,1) + 7   = 2*211 - 121 + 7      = 308
#:   (3, 4, 5): the constant alone                                     = 5
#:   (6, 6, 6): phi(5,6,6)/2 + phi(6,5,6)/4 + phi(6,6,5)/8
#:              = 283 + 164 + 83.125                                   = 530.125
THREE_ROWS = [
    ((1, 1, 1), [((2, 1, 1), 2.0), ((1, 2, 1), -1.0)], [7.0]),
    ((3, 4, 5), [], [5.0]),
    ((6, 6, 6), [((5, 6, 6), 0.5), ((6, 5, 6), 0.25), ((6, 6, 5), 0.125)], [0.0]),
]
THREE_ROWS_OVERWRITE = {(1, 1, 1): 308.0, (3, 4, 5): 5.0, (6, 6, 6): 530.125}


def _apply(out, phi, rows, mode=None, constant_scale=1.0, ncomp=1, grid_version=VERSION):
    """Build the table and run the band sweep — the production call path."""
    table = band_table(rows, VERSION)
    blockamr.apply_band_rows(
        out,
        phi,
        table,
        ncomp,
        blockamr.BandMode.Overwrite if mode is None else mode,
        constant_scale,
        grid_version,
    )
    return table


# ---------------------------------------------------------------------------
# the handle: what the table carries
# ---------------------------------------------------------------------------


def test_the_table_round_trips_every_property_it_was_built_from():
    """nrows, nbox, ncomp, stride, grid_version and the widest live row."""
    rows = _band_rows(
        [
            ((1, 1, 1), [((2, 1, 1), 1.0)], [1.0, 2.0]),
            ((2, 2, 2), [], [3.0, 4.0]),
            ((5, 5, 5), [((5, 5, 4), 1.0), ((5, 5, 6), 1.0)], [5.0, 6.0]),
        ],
        stride=4,
        box_offset=[0, 2, 3],
    )

    table = band_table(rows, 7)

    assert table.nrows == 3
    assert table.nbox == 2
    assert table.ncomp == 2
    assert table.stride == 4
    assert table.grid_version == 7
    assert table.max_nnz == 2


def test_the_patch_id_of_every_row_is_carried_through_to_the_handle():
    """Nothing consumes it yet — per-patch forces are the first reader — so it
    would be the easiest field to drop and the most expensive to add back."""
    rows = _band_rows(THREE_ROWS, stride=4)
    patch = np.array([4, 0, 4], dtype=np.int32)
    rows = BandRows(
        target=rows.target,
        stencil=rows.stencil,
        a=rows.a,
        nnz=rows.nnz,
        c=rows.c,
        patch=patch,
        box_offset=rows.box_offset,
        stride=rows.stride,
    )

    table = band_table(rows, VERSION)

    np.testing.assert_array_equal(table.patch, patch)


# ---------------------------------------------------------------------------
# the kernel: the row arithmetic
# ---------------------------------------------------------------------------


def test_a_row_is_its_live_stencil_plus_the_affine_constant():
    """The definition, on three hand-computed rows."""
    phi = _seed(_multifab(ngrow=1))
    out = _multifab(ngrow=0)

    _apply(out, phi, _band_rows(THREE_ROWS, stride=4))

    result = _values(out)[0]
    assert result[1, 1, 1, 0] == 308.0
    assert result[3, 4, 5, 0] == 5.0
    assert result[6, 6, 6, 0] == 530.125


def test_a_row_with_no_live_stencil_writes_the_constant_alone():
    """The non-fluid row: it reads nothing, so its poison slots prove it."""
    phi = _seed(_multifab(ngrow=1))
    out = _multifab(ngrow=0)

    _apply(out, phi, _band_rows([((3, 4, 5), [], [5.0])], stride=4))

    assert _values(out)[0][3, 4, 5, 0] == 5.0


def test_only_the_target_cells_are_written():
    """The band sweep touches the band and nothing else — the bulk the
    interior kernel wrote must survive it untouched."""
    phi = _seed(_multifab(ngrow=1))
    out = _multifab(ngrow=0)
    out.set_val(-1.0)

    _apply(out, phi, _band_rows(THREE_ROWS, stride=4))

    result = _values(out)[0]
    written = np.zeros(result.shape[:3], dtype=bool)
    for cell in THREE_ROWS_OVERWRITE:
        written[cell] = True
    np.testing.assert_array_equal(result[~written, 0], -1.0)


def test_overwrite_replaces_what_the_interior_sweep_wrote():
    phi = _seed(_multifab(ngrow=1))
    out = _multifab(ngrow=0)
    out.set_val(-1.0)

    _apply(out, phi, _band_rows(THREE_ROWS, stride=4), mode=blockamr.BandMode.Overwrite)

    result = _values(out)[0]
    assert result[1, 1, 1, 0] == 308.0
    assert result[3, 4, 5, 0] == 5.0
    assert result[6, 6, 6, 0] == 530.125


def test_add_accumulates_onto_what_the_interior_sweep_wrote():
    """The source-type method's mode: the same row, added instead of written."""
    phi = _seed(_multifab(ngrow=1))
    out = _multifab(ngrow=0)
    out.set_val(-1.0)

    _apply(out, phi, _band_rows(THREE_ROWS, stride=4), mode=blockamr.BandMode.Add)

    result = _values(out)[0]
    assert result[1, 1, 1, 0] == 307.0
    assert result[3, 4, 5, 0] == 4.0
    assert result[6, 6, 6, 0] == 529.125


def test_constant_scale_of_one_applies_the_whole_affine_constant():
    phi = _seed(_multifab(ngrow=1))
    out = _multifab(ngrow=0)

    _apply(out, phi, _band_rows(THREE_ROWS, stride=4), constant_scale=1.0)

    result = _values(out)[0]
    assert result[1, 1, 1, 0] == 308.0
    assert result[3, 4, 5, 0] == 5.0


def test_constant_scale_of_zero_leaves_the_linear_part_alone():
    """The Krylov matvec the implicit track needs: ``a . phi`` without the
    wall datum. Rows: 308 - 7 = 301, and the empty row collapses to 0."""
    phi = _seed(_multifab(ngrow=1))
    out = _multifab(ngrow=0)

    _apply(out, phi, _band_rows(THREE_ROWS, stride=4), constant_scale=0.0)

    result = _values(out)[0]
    assert result[1, 1, 1, 0] == 301.0
    assert result[3, 4, 5, 0] == 0.0
    assert result[6, 6, 6, 0] == 530.125


def test_constant_scale_multiplies_only_the_constant():
    """At 2.5: 301 + 2.5*7 = 318.5, and the empty row is 2.5*5 = 12.5."""
    phi = _seed(_multifab(ngrow=1))
    out = _multifab(ngrow=0)

    _apply(out, phi, _band_rows(THREE_ROWS, stride=4), constant_scale=2.5)

    result = _values(out)[0]
    assert result[1, 1, 1, 0] == 318.5
    assert result[3, 4, 5, 0] == 12.5


def test_every_component_gets_its_own_constant():
    """ncomp > 1: one row, one stencil, three components.

    phi carries +1000 per component, so the linear part is the same 90 in each
    and the three constants separate them: 97, 98, 99.
    """
    phi = _seed(_multifab(ncomp=3, ngrow=1))
    out = _multifab(ncomp=3, ngrow=0)
    rows = _band_rows(
        [((2, 2, 2), [((3, 2, 2), 1.0), ((2, 3, 2), -1.0)], [7.0, 8.0, 9.0])], stride=4
    )

    _apply(out, phi, rows, ncomp=3)

    result = _values(out)[0]
    assert result[2, 2, 2, 0] == 97.0
    assert result[2, 2, 2, 1] == 98.0
    assert result[2, 2, 2, 2] == 99.0


def test_the_stencil_stride_is_a_runtime_property_of_the_table():
    """The same rows at stride 4 and stride 9 give bitwise the same field —
    the stride is the boundary scheme's, not a compile-time constant."""
    phi = _seed(_multifab(ngrow=1))
    narrow = _multifab(ngrow=0)
    wide = _multifab(ngrow=0)

    _apply(narrow, phi, _band_rows(THREE_ROWS, stride=4))
    _apply(wide, phi, _band_rows(THREE_ROWS, stride=9))

    assert band_table(_band_rows(THREE_ROWS, stride=9), VERSION).stride == 9
    np.testing.assert_array_equal(_values(wide)[0], _values(narrow)[0])
    assert _values(wide)[0][6, 6, 6, 0] == 530.125


def test_an_empty_table_writes_nothing():
    """A body outside the domain produces one, and it must cost nothing."""
    phi = _seed(_multifab(ngrow=1))
    out = _multifab(ngrow=0)
    out.set_val(-1.0)
    rows = BandRows(
        target=np.zeros((0, 3), dtype=np.int32),
        stencil=np.zeros((0, 4, 3), dtype=np.int32),
        a=np.zeros((0, 4), dtype=np.float64),
        nnz=np.zeros(0, dtype=np.int32),
        c=np.zeros((0, 1), dtype=np.float64),
        patch=np.zeros(0, dtype=np.int32),
        box_offset=np.array([0, 0], dtype=np.int32),
        stride=4,
    )

    _apply(out, phi, rows)

    np.testing.assert_array_equal(_values(out)[0], -1.0)


# ---------------------------------------------------------------------------
# two MultiFabs: purity, determinism, and the in-place carve-out
# ---------------------------------------------------------------------------


def test_the_kernel_leaves_phi_bitwise_unchanged():
    """phi is read, out is written, and they are different MultiFabs — which
    is what removes the disjointness invariant the old design needed."""
    phi = _seed(_multifab(ngrow=1))
    out = _multifab(ngrow=0)
    before = _values(phi)[0].copy()

    _apply(out, phi, _band_rows(THREE_ROWS, stride=4))

    np.testing.assert_array_equal(_values(phi)[0], before)


def test_two_applies_of_the_same_table_are_bitwise_equal():
    phi = _seed(_multifab(ngrow=1))
    first = _multifab(ngrow=0)
    second = _multifab(ngrow=0)

    _apply(first, phi, _band_rows(THREE_ROWS, stride=4))
    _apply(second, phi, _band_rows(THREE_ROWS, stride=4))

    np.testing.assert_array_equal(_values(second)[0], _values(first)[0])


def test_an_in_place_apply_is_accepted_when_no_row_reads_a_cell():
    """The non-fluid pin: rows with an empty stencil have no in-place read, so
    the field may be passed as both source and destination — which is how a
    device-resident field gets pinned without a second kernel."""
    field = _seed(_multifab(ngrow=1))
    rows = _band_rows([((2, 2, 2), [], [0.0]), ((2, 2, 3), [], [0.0])], stride=1)

    _apply(field, field, rows)

    result = _values(field)[0]
    assert result[2, 2, 2, 0] == 0.0
    assert result[2, 2, 3, 0] == 0.0
    # 100*2 + 10*2 + 4 — the neighbour the pin must not have touched
    assert result[2, 2, 4, 0] == 224.0


def test_an_in_place_apply_is_rejected_when_a_row_reads_a_cell():
    """The two-fabs rule, and it must fire on the *reads*, not on the pointer."""
    field = _seed(_multifab(ngrow=1))
    rows = _band_rows([((2, 2, 2), [((2, 2, 3), 1.0)], [0.0])], stride=1)

    with pytest.raises(RuntimeError, match="same MultiFab"):
        _apply(field, field, rows)


# ---------------------------------------------------------------------------
# the CSR box grouping, in MFIterator order
# ---------------------------------------------------------------------------


def test_the_local_boxes_are_visited_in_box_array_order():
    """The convention box_offset is written against, asserted rather than
    assumed: run ``i`` of the table belongs to the ``i``-th box the iterator
    yields."""
    mf = _multifab(_split_boxes(), ngrow=1)

    corners = [mfi.valid_box().small_end() for mfi in blockamr.MFIterator(mf)]

    assert corners == [[0, 0, 0], [4, 0, 0]]


def test_box_offset_slices_the_rows_per_local_box():
    """One row per box, and each lands in its own box."""
    boxes = _split_boxes()
    phi = _seed(_multifab(boxes, ngrow=1))
    out = _multifab(boxes, ngrow=0)
    rows = _band_rows(
        [((1, 1, 1), [], [11.0]), ((5, 1, 1), [], [22.0])], stride=1, box_offset=[0, 1, 2]
    )

    _apply(out, phi, rows)

    first, second = _values(out)
    assert first[1, 1, 1, 0] == 11.0
    # the second box starts at i = 4, so global (5, 1, 1) is local (1, 1, 1)
    assert second[1, 1, 1, 0] == 22.0


def test_a_row_in_the_wrong_boxes_run_raises():
    """Rows grouped out of MFIterator order write another box's memory, so the
    per-box target range is checked before a single row is applied."""
    boxes = _split_boxes()
    phi = _seed(_multifab(boxes, ngrow=1))
    out = _multifab(boxes, ngrow=0)
    rows = _band_rows(
        [((1, 1, 1), [], [11.0]), ((5, 1, 1), [], [22.0])], stride=1, box_offset=[0, 2, 2]
    )

    with pytest.raises(RuntimeError, match="valid box"):
        _apply(out, phi, rows)


def test_a_row_may_read_a_neighbours_ghost_cell():
    """A band cell at a box edge reads across the box boundary, which is what
    the ghost region is for: phi(4, 1, 1) belongs to the second box and
    reaches the first through fill_boundary."""
    boxes = _split_boxes()
    geom = blockamr.Geometry(_full_box(), blockamr.RealBox([0.0] * 3, [1.0] * 3), 0, [1, 1, 1])
    phi = _seed(_multifab(boxes, ngrow=1))
    phi.fill_boundary(geom)
    out = _multifab(boxes, ngrow=0)
    rows = _band_rows([((3, 1, 1), [((4, 1, 1), 1.0)], [0.0])], stride=1, box_offset=[0, 1, 1])

    _apply(out, phi, rows)

    # 100*4 + 10*1 + 1
    assert _values(out)[0][3, 1, 1, 0] == 411.0


# ---------------------------------------------------------------------------
# the guards: a wrong table raises instead of computing plausible numbers
# ---------------------------------------------------------------------------


def test_a_stale_grid_version_raises_naming_both_versions():
    """A table that outlived its geometry indexes the wrong cells, and the
    numbers it produces are plausible — so this is the backstop that must not
    be silent."""
    phi = _seed(_multifab(ngrow=1))
    out = _multifab(ngrow=0)
    table = band_table(_band_rows(THREE_ROWS, stride=4), VERSION)

    with pytest.raises(RuntimeError) as excinfo:
        blockamr.apply_band_rows(out, phi, table, 1, blockamr.BandMode.Overwrite, 1.0, VERSION + 1)

    message = str(excinfo.value)
    assert f"grid_version {VERSION}" in message
    assert f"grid_version {VERSION + 1}" in message


def test_a_target_outside_the_valid_box_raises():
    """Targets are written, so an out-of-range one corrupts memory. The rule is
    the *valid* box, not the allocation: cell (8, 0, 0) is inside this fab's
    ghost region and still refused."""
    phi = _seed(_multifab(ngrow=1))
    out = _multifab(ngrow=1)
    rows = _band_rows([((N, 0, 0), [], [1.0])], stride=1)

    with pytest.raises(RuntimeError, match="valid box"):
        _apply(out, phi, rows)


def test_a_stencil_outside_the_fab_box_raises_naming_the_ghost_width():
    """The field's ghost width is what has to be big enough, and the message
    says by how much: a stencil at i = -2 needs 2, this field has 1."""
    phi = _seed(_multifab(ngrow=1))
    out = _multifab(ngrow=0)
    rows = _band_rows([((0, 0, 0), [((-2, 0, 0), 1.0)], [0.0])], stride=1)

    with pytest.raises(RuntimeError) as excinfo:
        _apply(out, phi, rows)

    message = str(excinfo.value)
    assert "ghost width of at least 2" in message
    assert "has 1" in message


# ---------------------------------------------------------------------------
# the device path
# ---------------------------------------------------------------------------


def test_the_sweep_runs_on_device_resident_multifabs():
    """The primary backend's fabs live in device memory and the table lives in
    a device vector; nothing in the band path stages a field through the host.
    On a CPU build the device arena is host memory and this still exercises the
    same launch."""
    phi = _seed(_multifab(ngrow=1, memory="device"))
    out = _multifab(ngrow=0, memory="device")

    _apply(out, phi, _band_rows(THREE_ROWS, stride=4))

    result = _values(out)[0]
    assert result[1, 1, 1, 0] == 308.0
    assert result[3, 4, 5, 0] == 5.0
    assert result[6, 6, 6, 0] == 530.125


# ---------------------------------------------------------------------------
# BandRows: the host-side validation, before anything is uploaded
# ---------------------------------------------------------------------------


def test_band_rows_rejects_a_target_that_is_not_int32():
    rows = _band_rows(THREE_ROWS, stride=4)

    with pytest.raises(ValueError, match="'target' must have dtype int32"):
        BandRows(
            target=rows.target.astype(np.int64),
            stencil=rows.stencil,
            a=rows.a,
            nnz=rows.nnz,
            c=rows.c,
            patch=rows.patch,
            box_offset=rows.box_offset,
            stride=rows.stride,
        )


def test_band_rows_rejects_a_stencil_whose_second_axis_is_not_the_stride():
    rows = _band_rows(THREE_ROWS, stride=4)

    with pytest.raises(ValueError, match="'stencil' must have shape"):
        BandRows(
            target=rows.target,
            stencil=np.ascontiguousarray(rows.stencil[:, :2]),
            a=rows.a,
            nnz=rows.nnz,
            c=rows.c,
            patch=rows.patch,
            box_offset=rows.box_offset,
            stride=rows.stride,
        )


def test_band_rows_rejects_a_live_length_above_the_stride():
    """The kernel indexes ``a`` and ``stencil`` without bounds checks, so a row
    claiming more live entries than it has slots is caught here."""
    rows = _band_rows(THREE_ROWS, stride=4)
    nnz = rows.nnz.copy()
    nnz[1] = 5

    with pytest.raises(ValueError, match="0 <= nnz <= stride = 4"):
        BandRows(
            target=rows.target,
            stencil=rows.stencil,
            a=rows.a,
            nnz=nnz,
            c=rows.c,
            patch=rows.patch,
            box_offset=rows.box_offset,
            stride=rows.stride,
        )


def test_band_rows_rejects_a_box_offset_that_does_not_end_at_the_row_count():
    rows = _band_rows(THREE_ROWS, stride=4)

    with pytest.raises(ValueError, match="must end at nrows = 3"):
        BandRows(
            target=rows.target,
            stencil=rows.stencil,
            a=rows.a,
            nnz=rows.nnz,
            c=rows.c,
            patch=rows.patch,
            box_offset=np.array([0, 2], dtype=np.int32),
            stride=rows.stride,
        )


def test_band_rows_rejects_a_constant_with_no_components():
    rows = _band_rows(THREE_ROWS, stride=4)

    with pytest.raises(ValueError, match="ncomp >= 1"):
        BandRows(
            target=rows.target,
            stencil=rows.stencil,
            a=rows.a,
            nnz=rows.nnz,
            c=np.zeros((3, 0), dtype=np.float64),
            patch=rows.patch,
            box_offset=rows.box_offset,
            stride=rows.stride,
        )
