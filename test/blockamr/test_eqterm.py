# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import blockamr
import numpy as np
import pytest
from blockamr.field import CellField
from blockamr.mesh import Mesh
from blockamr.dsl import exp, imp, EqTerm, Equation
from blockamr.dsl.exp import CellDivergence
from blockamr.operators.div import build_face_fluxes


def _make_mesh(n_cell=16, max_size=16):
    """Create a periodic Mesh on [0,1]^3."""
    box = blockamr.Box([0, 0, 0], [n_cell - 1, n_cell - 1, n_cell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    mesh = Mesh(ba, dm, geom)
    return mesh, box, dm, geom


def _zero_vel(x, y, z, t):
    return np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)


def _make_named_fluxes(box, dm, geom, name="phi"):
    ff = build_face_fluxes(_zero_vel, box, dm, geom, ngrow=1, t=0.0)
    ff.name = name
    return ff


def _gamma_one(x, y, z, t):
    return np.ones_like(x)


def test_composition_is_lazy():
    """+/- build an Equation holding the term objects; nothing evaluates."""
    mesh, box, dm, geom = _make_mesh()
    U = CellField(mesh, ncomp=1, ngrow=1, name="U")
    ff = _make_named_fluxes(box, dm, geom)

    ddt_term = exp.ddt(U)
    div_term = exp.div(ff, U)
    eqn = ddt_term + div_term - exp.laplacian(_gamma_one, U)

    assert isinstance(eqn, Equation)
    assert len(eqn.temporal_ops) == 1
    assert len(eqn.spatial_ops) == 2
    # The added terms are held as-is (no evaluation, no copies for +)
    assert eqn.temporal_ops[0] is ddt_term
    assert eqn.spatial_ops[0] is div_term
    assert eqn.implicit_lhs is None


def test_composition_does_not_mutate_operands():
    """Composing terms leaves the original term and equation objects unchanged."""
    mesh, box, dm, geom = _make_mesh()
    U = CellField(mesh, ncomp=1, ngrow=1, name="U")
    ff = _make_named_fluxes(box, dm, geom)

    div_term = exp.div(ff, U)
    eqn = exp.ddt(U) + div_term

    eqn2 = eqn - div_term
    assert div_term.coeff == 1.0  # subtraction scaled a copy
    assert len(eqn.spatial_ops) == 1  # original equation untouched
    assert len(eqn2.spatial_ops) == 2
    assert eqn2 is not eqn

    eqn3 = eqn + exp.laplacian(_gamma_one, U)
    assert len(eqn.spatial_ops) == 1
    assert len(eqn3.spatial_ops) == 2


def test_scalar_scaling_returns_new_term():
    """2 * term, term * 2 and -term return scaled copies, original unchanged."""
    mesh, box, dm, geom = _make_mesh()
    U = CellField(mesh, ncomp=1, ngrow=1, name="U")
    ff = _make_named_fluxes(box, dm, geom)
    div_term = exp.div(ff, U)

    doubled = 2.0 * div_term
    assert doubled is not div_term
    assert doubled.coeff == 2.0
    assert doubled.field is div_term.field
    assert div_term.coeff == 1.0

    tripled = div_term * 3.0
    assert tripled.coeff == 3.0

    negated = -div_term
    assert negated is not div_term
    assert negated.coeff == -1.0
    assert div_term.coeff == 1.0


def test_scheme_keys():
    """Terms derive OpenFOAM-style scheme keys from their operand names."""
    mesh, box, dm, geom = _make_mesh()
    U = CellField(mesh, ncomp=1, ngrow=1, name="U")
    p = CellField(mesh, ncomp=1, ngrow=1, name="p")
    ff = _make_named_fluxes(box, dm, geom, name="phi")

    assert exp.ddt(U).scheme_key == "ddt"
    assert exp.div(ff, U).scheme_key == "div(phi,U)"
    assert exp.laplacian(_gamma_one, U).scheme_key == "laplacian"
    assert exp.grad(p).scheme_key == "grad"
    assert exp.source(_gamma_one, U).scheme_key == "source"
    assert imp.laplacian(0.1, p).scheme_key == "laplacian"
    assert exp.div(U).scheme_key == "div(U)"


def test_scheme_key_unnamed_field_raises():
    """A div term over unnamed fields cannot form its key: clear ValueError."""
    mesh, box, dm, geom = _make_mesh()
    U = CellField(mesh, ncomp=1, ngrow=1)  # no name
    ff = build_face_fluxes(_zero_vel, box, dm, geom, ngrow=1, t=0.0)  # no name

    with pytest.raises(ValueError, match="no name"):
        _ = exp.div(ff, U).scheme_key


def test_eq_builds_implicit_equation():
    """imp.laplacian(dt, p) == exp.div(U) builds an implicit Equation."""
    mesh, box, dm, geom = _make_mesh()
    U = CellField(mesh, ncomp=3, ngrow=1, name="U")
    p = CellField(mesh, ncomp=1, ngrow=1, name="p")

    lhs = imp.laplacian(0.1, p)
    eqn = lhs == exp.div(U)

    assert isinstance(eqn, Equation)
    assert eqn.implicit_lhs is lhs
    assert isinstance(eqn.rhs, CellDivergence)
    assert eqn.rhs.vel_field is U
    assert eqn.explicit_terms == []


def test_hash_is_identity_based():
    """__eq__ builds equations, so hashing stays identity-based."""
    mesh, box, dm, geom = _make_mesh()
    U = CellField(mesh, ncomp=1, ngrow=1, name="U")

    term = exp.ddt(U)
    assert hash(term) == object.__hash__(term)
    # == does NOT compare — it builds an Equation
    assert isinstance(term == term, Equation)
    assert isinstance(term, EqTerm)
