# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import neon.blockamr as blockamr
from neon.blockamr.mesh import Mesh


def test_mesh_properties(blockamr_session):
    """Mesh wraps ba, dm, geom for single-level access."""
    box = blockamr.Box([0, 0, 0], [31, 31, 31])
    ba = blockamr.BoxArray(box)
    ba.max_size(32)
    dm = blockamr.DistributionMapping(ba)
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    mesh = Mesh(ba, dm, geom)
    assert mesh.n_levels() == 1
    assert mesh.finest_level() == 0
    assert mesh.max_level == 0
    assert mesh.geom(0) is geom
    assert mesh.box_array(0) is ba
    assert mesh.dm(0) is dm


def test_mesh_register_field_triggers_on_new_level(blockamr_session):
    """register_field immediately triggers _on_new_level(0, ba, dm)."""
    box = blockamr.Box([0, 0, 0], [31, 31, 31])
    ba = blockamr.BoxArray(box)
    ba.max_size(32)
    dm = blockamr.DistributionMapping(ba)
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    mesh = Mesh(ba, dm, geom)

    calls = []

    class FakeField:
        def _on_new_level(self, lev, ba, dm):
            calls.append(lev)

    mesh.register_field(FakeField())
    assert calls == [0]
