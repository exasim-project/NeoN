# SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import pytest
import neon
from neon import exp, imp


def test_dsl_scalar_operators():
    # Setup
    exec = neon.SerialExecutor()
    mesh = neon.create_1d_uniform_mesh(exec, 10)
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


def test_dsl_vector_operators():
    # Setup
    exec = neon.SerialExecutor()
    mesh = neon.create_1d_uniform_mesh(exec, 10)
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
