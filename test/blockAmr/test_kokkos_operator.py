# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Kokkos and AMReX must compute the same thing before either timing is believed.

The bench in ``src/blockAmr/bench/`` runs three cell kernels (axpy, a
7-point Laplacian, a VanLeer-limited divergence) through seven launchers. One
launch per box:

  ``amrex``         -- ``amrex::ParallelFor``, the baseline;
  ``kokkos_md``     -- ``Kokkos::MDRangePolicy<Rank<3>>``, idiomatic Kokkos;
  ``kokkos_flat``   -- ``Kokkos::RangePolicy`` + manual ijk decomposition, which is
                       AMReX's own scheme and the only form ``NeoN::parallelFor``
                       can express today (it takes a 1D range);
  ``kokkos_md_a4``  -- diagnostic: Kokkos launcher with AMReX's own ``Array4``
                       accessor, to separate launcher cost from accessor cost;
  ``kokkos_stream`` -- ``kokkos_md`` spread over as many Kokkos streams as AMReX
                       round-robins its box loop across.

And one launch for ALL boxes, which is where the multi-box launch cost goes away:

  ``amrex_fused``   -- ``amrex::ParallelFor(mf, f)``, AMReX's own fused path;
  ``kokkos_team``   -- ``Kokkos::TeamPolicy`` over the same block decomposition,
                       reading the same cached ``BoxIndexer`` table AMReX built.

All of them invoke ONE templated kernel body over the same MultiFab memory -- AMReX
through ``Array4``, Kokkos through an unmanaged ``View`` with the fab origin
subtracted -- so a disagreement here means a launcher indexes wrongly, which is the
one failure mode that would make a fast kernel worthless. The fused launchers add a
second such mode: they resolve the box on the device, so a wrong box-to-Array4
mapping would show up as data landing in the wrong box.

Every operator is checked against an independent numpy reference AND against the
other backends, single-box and multi-box, because a box boundary changes which
cells come from ``FillBoundary`` rather than the periodic wrap.
"""

import numpy as np
import pytest

import blockamr

BACKENDS = [
    "amrex",
    "amrex_fused",
    "kokkos_md",
    "kokkos_flat",
    "kokkos_md_a4",
    "kokkos_stream",
    "kokkos_team",
]

# Small on purpose: this file proves correctness, not throughput. The bench driver
# in benchmarks/blockAmr/ is where sizes that saturate bandwidth belong.
SHAPE = (16, 16, 16)
MULTIBOX = [None, 8]
MULTIBOX_IDS = ["1box", "multibox"]

DX = (1.0 / SHAPE[0], 1.0 / SHAPE[1], 1.0 / SHAPE[2])


def _mesh(max_size=None):
    box = blockamr.Box([0, 0, 0], [s - 1 for s in SHAPE])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(max(SHAPE) if max_size is None else max_size)
    dm = blockamr.DistributionMapping(ba)
    return geom, ba, dm


def _nboxes(mf):
    return sum(1 for _ in blockamr.MFIterator(mf))


def _scatter(mf, values):
    """Fill the valid region of a (possibly multi-box) MultiFab from a global array."""
    for mfi in blockamr.MFIterator(mf):
        bx = mfi.valid_box()
        s, b = bx.small_end(), bx.big_end()
        arr = mf.copy_to_host(mfi)
        arr[:, :, :, 0] = values[s[0] : b[0] + 1, s[1] : b[1] + 1, s[2] : b[2] + 1]
        mf.copy_from(mfi, arr)
    return mf


def _gather(mf, shape):
    out = np.full(shape, np.nan)
    for mfi in blockamr.MFIterator(mf):
        bx = mfi.valid_box()
        s, b = bx.small_end(), bx.big_end()
        arr = mf.copy_to_host(mfi)
        out[s[0] : b[0] + 1, s[1] : b[1] + 1, s[2] : b[2] + 1] = arr[:, :, :, 0]
    assert not np.isnan(out).any(), "gather left holes — boxes did not cover the domain"
    return out


def _cell_mf(ba, dm, geom, values, nghost):
    mf = blockamr.MultiFab(ba, dm, 1, nghost)
    mf.set_val(0.0)
    _scatter(mf, values)
    if nghost:
        # Periodic fill, done once here and never inside the timed region.
        mf.fill_boundary(geom)
    return mf


def _face_mf(ba, dm, d, values):
    typ = [0, 0, 0]
    typ[d] = 1
    fba = blockamr.convert_ba(ba, blockamr.IntVect(*typ))
    mf = blockamr.MultiFab(fba, dm, 1, 0)
    mf.set_val(0.0)
    for mfi in blockamr.MFIterator(mf):
        bx = mfi.valid_box()
        s, b = bx.small_end(), bx.big_end()
        arr = mf.copy_to_host(mfi)
        arr[:, :, :, 0] = values[s[0] : b[0] + 1, s[1] : b[1] + 1, s[2] : b[2] + 1]
        mf.copy_from(mfi, arr)
    return mf


# ---------------------------------------------------------------------------
# numpy references. np.roll supplies the periodic wrap, matching fill_boundary
# on an all-periodic domain.
# ---------------------------------------------------------------------------
def _ref_axpy(x, y, a):
    return a * x + y


def _ref_laplacian(phi):
    out = np.zeros_like(phi)
    for d, h in enumerate(DX):
        out += (np.roll(phi, -1, axis=d) + np.roll(phi, 1, axis=d) - 2.0 * phi) / h**2
    return out


def _vanleer_corr(d_up, d_down):
    prod = d_up * d_down
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(prod > 0.0, 2.0 * prod / (d_up + d_down), 0.0)
    return corr


def _ref_vanleer(phi, faces):
    """Mirror of divVanLeerCell: limited upwind reconstruction on both faces."""
    total = np.zeros_like(phi)
    for d, h in enumerate(DX):
        # faces[d] has shape[d] + 1 entries along d; fl is index 0..n-1, fr is 1..n
        sl_l = tuple(slice(0, -1) if a == d else slice(None) for a in range(3))
        sl_r = tuple(slice(1, None) if a == d else slice(None) for a in range(3))
        fl, fr = faces[d][sl_l], faces[d][sl_r]

        sm2 = np.roll(phi, 2, axis=d)
        sm1 = np.roll(phi, 1, axis=d)
        s0 = phi
        sp1 = np.roll(phi, -1, axis=d)
        sp2 = np.roll(phi, -2, axis=d)

        dl = s0 - sm1
        pl = np.where(
            fl >= 0.0,
            sm1 + 0.5 * _vanleer_corr(sm1 - sm2, dl),
            s0 - 0.5 * _vanleer_corr(sp1 - s0, dl),
        )
        dr = sp1 - s0
        pr = np.where(
            fr >= 0.0,
            s0 + 0.5 * _vanleer_corr(s0 - sm1, dr),
            sp1 - 0.5 * _vanleer_corr(sp2 - sp1, dr),
        )
        total += (fr * pr - fl * pl) / h
    return total


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def test_every_operator_is_registered():
    """Registration happens via explicit instantiation, which fails silently if the
    instantiation is dropped — so the table is asserted, not assumed."""
    got = set(blockamr.bench_operators())
    want = {f"{k}/{b}" for k in ("axpy", "laplacian", "vanleer") for b in BACKENDS}
    assert got == want, f"missing {want - got}, unexpected {got - want}"


def test_kokkos_is_initialized_by_the_runtime():
    assert blockamr.kokkos_available()
    assert blockamr.kokkos_execution_space()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("max_size", MULTIBOX, ids=MULTIBOX_IDS)
def test_axpy_matches_numpy(backend, max_size):
    geom, ba, dm = _mesh(max_size)
    rng = np.random.default_rng(0)
    x, y = rng.random(SHAPE), rng.random(SHAPE)
    a = 2.5

    in_mf = _cell_mf(ba, dm, geom, x, 0)
    out_mf = _cell_mf(ba, dm, geom, y, 0)
    if max_size is not None:
        assert _nboxes(out_mf) > 1, "max_size did not split the domain — test is vacuous"

    blockamr.apply_operator(f"axpy/{backend}", out_mf, in_mf, a=a)
    got = _gather(out_mf, SHAPE)
    # Not bit-exact against numpy: the device compiles with --use_fast_math and
    # contracts a*x + y into a single FMA, so it is one ULP MORE accurate than
    # numpy's separate multiply and add. Cross-backend equality is the exactness
    # claim that matters, and test_backends_agree_with_each_other makes it.
    np.testing.assert_allclose(got, _ref_axpy(x, y, a), rtol=1e-15, atol=0.0)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("max_size", MULTIBOX, ids=MULTIBOX_IDS)
def test_laplacian_matches_numpy(backend, max_size):
    geom, ba, dm = _mesh(max_size)
    rng = np.random.default_rng(1)
    phi = rng.random(SHAPE)

    in_mf = _cell_mf(ba, dm, geom, phi, 1)
    out_mf = _cell_mf(ba, dm, geom, np.zeros(SHAPE), 0)
    if max_size is not None:
        assert _nboxes(out_mf) > 1, "max_size did not split the domain — test is vacuous"

    blockamr.apply_operator(f"laplacian/{backend}", out_mf, in_mf, dx=DX[0], dy=DX[1], dz=DX[2])
    got = _gather(out_mf, SHAPE)
    np.testing.assert_allclose(got, _ref_laplacian(phi), rtol=1e-13, atol=1e-10)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("max_size", MULTIBOX, ids=MULTIBOX_IDS)
def test_vanleer_matches_numpy(backend, max_size):
    geom, ba, dm = _mesh(max_size)
    rng = np.random.default_rng(2)
    phi = rng.random(SHAPE)
    # Face velocities of both signs, so both branches of the limiter run.
    faces = [
        rng.random(tuple(s + (1 if a == d else 0) for a, s in enumerate(SHAPE))) - 0.5
        for d in range(3)
    ]

    in_mf = _cell_mf(ba, dm, geom, phi, 2)
    out_mf = _cell_mf(ba, dm, geom, np.zeros(SHAPE), 0)
    face_mfs = [_face_mf(ba, dm, d, faces[d]) for d in range(3)]
    if max_size is not None:
        assert _nboxes(out_mf) > 1, "max_size did not split the domain — test is vacuous"

    blockamr.apply_operator(
        f"vanleer/{backend}",
        out_mf,
        in_mf,
        fx=face_mfs[0],
        fy=face_mfs[1],
        fz=face_mfs[2],
        dx=DX[0],
        dy=DX[1],
        dz=DX[2],
    )
    got = _gather(out_mf, SHAPE)
    np.testing.assert_allclose(got, _ref_vanleer(phi, faces), rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("op", ["axpy", "laplacian", "vanleer"])
def test_backends_agree_with_each_other(op):
    """Cross-check independent of the numpy reference: every launcher must produce
    identical bytes, since they run the same kernel body. Multi-box on purpose, so
    the fused launchers' device-side box resolution is exercised too."""
    geom, ba, dm = _mesh(8)
    rng = np.random.default_rng(3)
    phi = rng.random(SHAPE)
    info = blockamr.bench_operator_info(f"{op}/amrex")

    faces = None
    if info["needs_faces"]:
        faces = [
            rng.random(tuple(s + (1 if a == d else 0) for a, s in enumerate(SHAPE))) - 0.5
            for d in range(3)
        ]

    results = {}
    for backend in BACKENDS:
        in_mf = _cell_mf(ba, dm, geom, phi, info["nghost"])
        out_mf = _cell_mf(ba, dm, geom, np.zeros(SHAPE), 0)
        kwargs = {}
        if faces is not None:
            fmfs = [_face_mf(ba, dm, d, faces[d]) for d in range(3)]
            kwargs = {"fx": fmfs[0], "fy": fmfs[1], "fz": fmfs[2]}
        blockamr.apply_operator(
            f"{op}/{backend}", out_mf, in_mf, dx=DX[0], dy=DX[1], dz=DX[2], **kwargs
        )
        results[backend] = _gather(out_mf, SHAPE)

    for backend in BACKENDS[1:]:
        np.testing.assert_array_equal(
            results[BACKENDS[0]],
            results[backend],
            err_msg=f"{op}: {backend} disagrees with {BACKENDS[0]}",
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_bench_operator_reports_a_rate(backend):
    """The timing path itself has to run — a few iterations, no performance claim."""
    geom, ba, dm = _mesh()
    rng = np.random.default_rng(4)
    in_mf = _cell_mf(ba, dm, geom, rng.random(SHAPE), 1)
    out_mf = _cell_mf(ba, dm, geom, np.zeros(SHAPE), 0)

    stats = dict(
        blockamr.bench_operator(
            f"laplacian/{backend}",
            out_mf,
            in_mf,
            dx=DX[0],
            dy=DX[1],
            dz=DX[2],
            iters=5,
            batches=3,
        )
    )
    assert stats["ncells"] == SHAPE[0] * SHAPE[1] * SHAPE[2]
    assert stats["nboxes"] == 1
    assert stats["ms_min"] > 0.0
    assert stats["ms_median"] >= stats["ms_min"]
    assert stats["gb_per_s"] > 0.0
