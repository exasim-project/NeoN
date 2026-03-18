# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import blockamr
from blockamr.field import Field
from blockamr.operators.ddt import Ddt


def _make_field(n_cell=64, max_size=32, ngrow=1, name="phi"):
    """Create a periodic Field wrapping a MultiFab + Geometry."""
    box = blockamr.Box([0, 0, 0], [n_cell - 1, n_cell - 1, n_cell - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(max_size)
    dm = blockamr.DistributionMapping(ba)
    mf = blockamr.MultiFab(ba, dm, 1, ngrow)
    return Field(mf, geom, name=name)


def test_ddt_stores_field():
    """Ddt operator stores reference to its field."""
    field = _make_field()
    op = Ddt(field)
    assert op.field is field
    assert op.coeff == 1.0
