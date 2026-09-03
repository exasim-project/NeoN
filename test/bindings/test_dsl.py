# SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import pytest
import neon
from neon import exp, imp


def test_dsl_scalar_operators(executor):
    # Setup
    name, exec = executor
    mesh = neon.create_1d_uniform_mesh(exec, 10, 1.0)
    phi = neon.ScalarVolumeField(exec, "phi", mesh)

    # 1. Test Temporal Operator Construction
    ddt_op = imp.ddt(phi)
    assert isinstance(ddt_op, neon.TemporalOperatorScalar)
    assert ddt_op.get_name() == "DdtOperator"

    # 2. Test Spatial Operator Construction
    source_op = imp.source(phi, phi)
    assert isinstance(source_op, neon.SpatialOperatorScalar)
    assert source_op.get_name() == "sourceTerm"

    # spataial + spatial -> expression
    eqn = source_op + source_op
    assert isinstance(eqn, neon.ExpressionScalar)
    assert eqn.size() == 2

    # spataial - spatial -> expression
    eqn = source_op - source_op
    assert isinstance(eqn, neon.ExpressionScalar)
    assert eqn.size() == 2

    # temporal + temporal -> expression
    eqn = ddt_op + ddt_op
    assert isinstance(eqn, neon.ExpressionScalar)
    assert eqn.size() == 2

    # temporal - temporal -> expression
    eqn = ddt_op - ddt_op
    assert isinstance(eqn, neon.ExpressionScalar)
    assert eqn.size() == 2

    # temporal + spatial -> expression
    eqn = ddt_op + source_op
    assert eqn.size() == 2

    # temporal - spatial -> expression
    eqn = ddt_op - source_op
    assert eqn.size() == 2

    # spatial + temporal -> expression
    eqn = source_op + ddt_op
    assert eqn.size() == 2

    # spatial - temporal -> expression
    eqn = source_op - ddt_op
    assert eqn.size() == 2

    eqn = eqn + ddt_op
    assert eqn.size() == 3

    eqn = ddt_op + ddt_op + source_op + source_op
    assert eqn.size() == 4

    scaled_eqn = 2.0 * eqn
    assert isinstance(scaled_eqn, neon.ExpressionScalar)
    assert scaled_eqn.size() == 4

    eqn1 = ddt_op + source_op
    eqn2 = source_op + source_op
    res_eqn = eqn1 + eqn2
    assert isinstance(res_eqn, neon.ExpressionScalar)
    assert res_eqn.size() == 4

    res_eqn = eqn1 - eqn2
    assert isinstance(res_eqn, neon.ExpressionScalar)
    assert res_eqn.size() == 4


def test_dsl_vector_operators(executor):
    # Setup
    name, exec = executor
    mesh = neon.create_1d_uniform_mesh(exec, 10, 1.0)
    phi = neon.VectorVolumeField(exec, "phi", mesh)
    coeff = neon.ScalarVolumeField(exec, "coeff", mesh)

    # Temporal Operator Construction
    ddt_op = imp.ddt(phi)
    assert isinstance(ddt_op, neon.TemporalOperatorVector)
    assert ddt_op.get_name() == "DdtOperator"

    # Spatial Operator Construction
    source_op = imp.source(coeff, phi)
    assert isinstance(source_op, neon.SpatialOperatorVector)
    assert source_op.get_name() == "sourceTerm"

    # spataial + spatial -> expression
    eqn = source_op + source_op
    assert isinstance(eqn, neon.ExpressionVector)
    assert eqn.size() == 2

    # spataial - spatial -> expression
    eqn = source_op - source_op
    assert isinstance(eqn, neon.ExpressionVector)
    assert eqn.size() == 2

    # temporal + temporal -> expression
    eqn = ddt_op + ddt_op
    assert isinstance(eqn, neon.ExpressionVector)
    assert eqn.size() == 2

    # temporal - temporal -> expression
    eqn = ddt_op - ddt_op
    assert isinstance(eqn, neon.ExpressionVector)
    assert eqn.size() == 2

    # temporal + spatial -> expression
    eqn = ddt_op + source_op
    assert eqn.size() == 2

    # temporal - spatial -> expression
    eqn = ddt_op - source_op
    assert eqn.size() == 2

    # spatial + temporal -> expression
    eqn = source_op + ddt_op
    assert eqn.size() == 2

    # spatial - temporal -> expression
    eqn = source_op - ddt_op
    assert eqn.size() == 2

    eqn = eqn + ddt_op
    assert eqn.size() == 3

    eqn = ddt_op + ddt_op + source_op + source_op
    assert eqn.size() == 4

    scaled_eqn = 2.0 * eqn
    assert isinstance(scaled_eqn, neon.ExpressionVector)
    assert scaled_eqn.size() == 4

    eqn1 = ddt_op + source_op
    eqn2 = source_op + source_op
    res_eqn = eqn1 + eqn2
    assert isinstance(res_eqn, neon.ExpressionVector)
    assert res_eqn.size() == 4

    res_eqn = eqn1 - eqn2
    assert isinstance(res_eqn, neon.ExpressionVector)
    assert res_eqn.size() == 4


def _cell_volumes(mesh):
    """The mesh cell volumes as a host NumPy array."""
    np = pytest.importorskip("numpy")
    return np.asarray(mesh.cell_volumes.copy_to_host())


def test_dsl_susp_scalar_sign_split(executor):
    """imp.susp assembles OpenFOAM's SuSp sign split for a scalar field.

    coeff >= 0 -> implicit: diagonal += coeff*V, rhs untouched.
    coeff <  0 -> explicit: diagonal untouched, rhs -= coeff*V*phi (i.e. += |coeff|*V*phi).

    Both branches share the same operator name and Python type, so only the assembled
    system distinguishes them (and distinguishes imp.susp from imp.source).
    """
    np = pytest.importorskip("numpy")
    name, exec = executor
    mesh = neon.create_1d_uniform_mesh(exec, 10, 1.0)
    vol = _cell_volumes(mesh)

    phi = neon.ScalarVolumeField(exec, "phi", mesh)
    neon.fill(phi.internal_vector(), 10.0)
    coeff = neon.ScalarVolumeField(exec, "coeff", mesh)

    # Positive coefficient: behaves like imp.source (Sp).
    neon.fill(coeff.internal_vector(), 2.0)
    op = imp.susp(coeff, phi)
    assert isinstance(op, neon.SpatialOperatorScalar)
    values, rhs = neon.assemble_spatial(op + op, mesh)
    # Two identical terms, so the total diagonal is 2 * (2 * V).
    assert np.isclose(np.sum(values), 2.0 * np.sum(2.0 * vol))
    assert np.allclose(rhs, 0.0)

    # Negative coefficient: nothing implicit, everything on the rhs.
    neon.fill(coeff.internal_vector(), -3.0)
    values, rhs = neon.assemble_spatial(imp.susp(coeff, phi) + imp.susp(coeff, phi), mesh)
    assert np.allclose(values, 0.0)
    assert np.allclose(rhs, 2.0 * 3.0 * vol * 10.0)

    # Contrast: imp.source is unconditionally implicit, so a negative coefficient lands
    # on the diagonal instead. This is what an incorrectly bound imp.susp would silently do.
    values, rhs = neon.assemble_spatial(imp.source(coeff, phi) + imp.source(coeff, phi), mesh)
    assert np.isclose(np.sum(values), 2.0 * np.sum(-3.0 * vol))
    assert np.allclose(rhs, 0.0)


def test_dsl_susp_vector_sign_split(executor):
    """imp.susp resolves and assembles for a Vec3 field (the <Vec3> overload)."""
    np = pytest.importorskip("numpy")
    name, exec = executor
    mesh = neon.create_1d_uniform_mesh(exec, 10, 1.0)
    vol = _cell_volumes(mesh)

    phi = neon.VectorVolumeField(exec, "U", mesh)
    neon.fill(phi.internal_vector(), neon.Vec3(10.0, 20.0, 30.0))
    coeff = neon.ScalarVolumeField(exec, "coeff", mesh)

    neon.fill(coeff.internal_vector(), 2.0)
    op = imp.susp(coeff, phi)
    assert isinstance(op, neon.SpatialOperatorVector)
    values, rhs = neon.assemble_spatial(op + op, mesh)
    assert np.isclose(sum(v[0] for v in values), np.sum(2.0 * 2.0 * vol))
    assert all(r[0] == 0.0 and r[1] == 0.0 and r[2] == 0.0 for r in rhs)

    neon.fill(coeff.internal_vector(), -3.0)
    values, rhs = neon.assemble_spatial(imp.susp(coeff, phi) + imp.susp(coeff, phi), mesh)
    assert all(v[0] == 0.0 and v[1] == 0.0 and v[2] == 0.0 for v in values)
    assert np.allclose([r[0] for r in rhs], 2.0 * 3.0 * vol * 10.0)
    assert np.allclose([r[1] for r in rhs], 2.0 * 3.0 * vol * 20.0)
    assert np.allclose([r[2] for r in rhs], 2.0 * 3.0 * vol * 30.0)
