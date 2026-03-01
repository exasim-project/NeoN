# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import pytest
import neon

if not hasattr(neon, "partition_mesh"):
    pytest.skip("neon built without METIS support", allow_module_level=True)


def test_partition_bindings_exist():
    assert hasattr(neon, "partition_mesh")
    assert hasattr(neon, "extract_sub_mesh")


def test_partition_mesh_single_part():
    exec_ = neon.SerialExecutor()
    mesh = neon.create_uniform_2d_mesh(exec_, 4, 4)
    cell_part = neon.partition_mesh(mesh, 1)
    assert len(cell_part) == mesh.n_cells()
    assert all(p == 0 for p in cell_part)


def test_partition_mesh_four_parts_balanced():
    exec_ = neon.SerialExecutor()
    mesh = neon.create_uniform_2d_mesh(exec_, 4, 4)  # 16 cells
    cell_part = neon.partition_mesh(mesh, 4)
    assert len(cell_part) == 16
    assert all(0 <= p < 4 for p in cell_part)
    counts = [cell_part.count(p) for p in range(4)]
    for c in counts:
        assert 2 <= c <= 8  # METIS should be roughly balanced


def test_extract_sub_mesh_total_cells_preserved():
    exec_ = neon.SerialExecutor()
    mesh = neon.create_uniform_2d_mesh(exec_, 4, 4)
    cell_part = neon.partition_mesh(mesh, 4)
    total = sum(neon.extract_sub_mesh(mesh, cell_part, p).n_cells() for p in range(4))
    assert total == mesh.n_cells()
