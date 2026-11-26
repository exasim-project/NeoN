# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import pytest

import neon


def _calculated_dict():
    d = neon.Dictionary()
    d.insert_string("type", "calculated")
    return d


def test_boundary_context_drives_inlet_outlet(executor):
    """BoundaryContext threaded through correct_boundary_conditions(ctx) actually
    switches a flux-dependent BC, the same way the C++ volInletOutlet.cpp test does --
    proof the Python binding isn't a no-op passthrough. Without this, every
    context-aware BC (inletOutlet, totalPressure, waveTransmissive,
    pressureInletOutletVelocity, a sine-forced fixedValue, ...) is unreachable from
    Python and silently falls back to its no-context default.
    """
    np = pytest.importorskip("numpy")
    name, exec = executor
    mesh = neon.create_single_cell_mesh(exec)

    io_dict = neon.Dictionary()
    io_dict.insert_string("type", "inletOutlet")
    io_dict.insert_double("inletValue", 7.0)

    bcs = [neon.ScalarVolumeBoundary(mesh, io_dict, 0)]
    bcs += [neon.ScalarVolumeBoundary(mesh, _calculated_dict(), p) for p in range(1, 4)]
    field = neon.ScalarVolumeField(exec, "f", mesh, bcs)
    neon.fill(field.internal_vector(), 4.0)

    # phi is the face flux: a SURFACE field, not a volume one -- inletOutlet's kernel
    # looks it up via ctx.hasSurfaceScalar()/surfaceScalarField(), so inserting it as a
    # volume field would silently miss and degrade to the no-context fallback.
    phi = neon.ScalarSurfaceField(exec, "phi", mesh)

    def patch0_value(flux):
        neon.fill(phi.internal_vector(), flux)
        neon.fill(phi.boundary_data_value(), flux)

        ctx = neon.BoundaryContext()
        ctx.insert("phi", phi)
        field.correct_boundary_conditions(ctx)

        return np.asarray(field.boundary_data_value().copy_to_host())[0]

    # inflow (phi < 0): Dirichlet at inletValue
    assert patch0_value(-2.0) == pytest.approx(7.0)
    # outflow (phi >= 0): zero-gradient (owner-cell value, 4.0)
    assert patch0_value(2.0) == pytest.approx(4.0)


def test_boundary_context_insert_value_is_accepted():
    """insert_value is plumbing with no consumer on this branch yet (a BC that reads
    a plain value, e.g. waveTransmissive's deltaT or a sine-forced fixedValue's time,
    lands separately) -- this only checks the binding itself accepts the call for
    several values/keys without raising. The value actually reaching a consuming BC
    is exercised the same way insert() is in test_boundary_context_drives_inlet_outlet
    above (same underlying map-insertion code path), and by each such BC's own C++
    Catch2 test."""
    ctx = neon.BoundaryContext()
    ctx.insert_value("deltaT", 0.5)
    ctx.insert_value("time", 1.25)
