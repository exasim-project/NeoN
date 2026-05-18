# SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import neon


def test_scalar_volume_field_and_bcs(executor):
    name, exec = executor
    mesh = neon.create_1d_uniform_mesh(exec, 4)

    bcs = neon.create_calculated_volume_bcs_scalar(mesh)
    assert len(bcs) == mesh.n_boundaries()

    field = neon.ScalarVolumeField(exec, "p", mesh)
    assert field.size() == mesh.n_cells()

    neon.fill(field.internal_vector(), 2.0)
    assert neon.equal(field.internal_vector(), 2.0)
