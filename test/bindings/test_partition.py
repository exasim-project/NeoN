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


def test_per_neighbor_proc_patch_names():
    exec_ = neon.SerialExecutor()
    mesh = neon.create_uniform_2d_mesh(exec_, 4, 4)
    cell_part = neon.partition_mesh(mesh, 4)

    for part_id in range(4):
        sub = neon.extract_sub_mesh(mesh, cell_part, part_id)
        patch_names = neon.get_patch_names(sub)
        proc_patches = [n for n in patch_names if n.startswith("proc")]
        assert len(proc_patches) > 0, f"Part {part_id} should have proc patches"
        for name in proc_patches:
            assert name.startswith(f"proc{part_id}to"), (
                f"Expected proc{part_id}to*, got {name}"
            )
            assert "procBoundary_" not in name


def test_global_cell_ids_available():
    exec_ = neon.SerialExecutor()
    mesh = neon.create_uniform_2d_mesh(exec_, 4, 4)
    cell_part = neon.partition_mesh(mesh, 4)

    all_ids = []
    for part_id in range(4):
        sub = neon.extract_sub_mesh(mesh, cell_part, part_id)
        ids = neon.get_global_cell_ids(sub)
        assert len(ids) == sub.n_cells()
        all_ids.extend(ids)

    # All global cell IDs across all parts should be exactly {0..15}
    assert sorted(all_ids) == list(range(16))


def test_ghost_cell_data_available():
    exec_ = neon.SerialExecutor()
    mesh = neon.create_uniform_2d_mesh(exec_, 4, 4)
    cell_part = neon.partition_mesh(mesh, 4)

    for part_id in range(4):
        sub = neon.extract_sub_mesh(mesh, cell_part, part_id)
        ghost_ids = neon.get_ghost_cell_ids(sub)
        # All ghost cells must belong to other partitions
        assert len(ghost_ids) > 0
        for gid in ghost_ids:
            assert cell_part[gid] != part_id


def test_write_vtm_with_ghost_cells(tmp_path):
    exec_ = neon.SerialExecutor()
    mesh = neon.create_uniform_2d_mesh(exec_, 4, 4)
    cell_part = neon.partition_mesh(mesh, 4)

    sub = neon.extract_sub_mesh(mesh, cell_part, 0)
    ghost_ids = neon.get_ghost_cell_ids(sub)
    n_ghost = len(ghost_ids)

    path = str(tmp_path / "ghost_partition_0.vtm")
    neon.write_vtm(sub, path, include_ghosts=True)

    import pyvista as pv

    grid = pv.read(path)
    internal = grid[0]
    assert internal.n_cells == sub.n_cells() + n_ghost

    # Verify ghostCells cell data array
    assert "ghostCells" in internal.cell_data
    ghost_flags = internal.cell_data["ghostCells"]
    assert sum(ghost_flags == 0) == sub.n_cells()
    assert sum(ghost_flags == 1) == n_ghost
