# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import blockamr


def test_index_type_default_is_cell_centered():
    """Default IndexType should be all cell-centred."""
    t = blockamr.IndexType()
    assert t.cell_centered()


def test_index_type_construct_nodal():
    """Construct a fully nodal IndexType."""
    NODE = blockamr.IndexType.CellIndex.NODE
    t = blockamr.IndexType(NODE, NODE, NODE)
    assert t.node_centered()
    assert not t.cell_centered()


def test_index_type_face_centred():
    """Construct x-face IndexType: nodal in x, cell-centred in y,z."""
    NODE = blockamr.IndexType.CellIndex.NODE
    CELL = blockamr.IndexType.CellIndex.CELL
    t = blockamr.IndexType(NODE, CELL, CELL)
    assert t.node_centered(0)
    assert t.cell_centered(1)
    assert t.cell_centered(2)
    assert not t.cell_centered()
    assert not t.node_centered()


def test_index_type_static_factories():
    """cell_type() and node_type() return the expected types."""
    cell = blockamr.IndexType.cell_type()
    node = blockamr.IndexType.node_type()
    assert cell.cell_centered()
    assert node.node_centered()


def test_index_type_repr():
    """Repr should show centering letters."""
    t = blockamr.IndexType()
    r = repr(t)
    assert "C,C,C" in r


def test_index_type_equality():
    """Two default IndexTypes should be equal."""
    a = blockamr.IndexType()
    b = blockamr.IndexType.cell_type()
    assert a == b


# --- Box centering tests ---


def test_box_with_index_type():
    """Construct a face-centred box via IndexType constructor."""
    NODE = blockamr.IndexType.CellIndex.NODE
    CELL = blockamr.IndexType.CellIndex.CELL
    xface = blockamr.IndexType(NODE, CELL, CELL)
    box = blockamr.Box([0, 0, 0], [63, 63, 63], xface)
    assert box.ix_type() == xface
    assert not box.cell_centered()


def test_box_surrounding_nodes_all():
    """surrounding_nodes() converts cell box to fully nodal."""
    box = blockamr.Box([0, 0, 0], [63, 63, 63])
    assert box.num_pts() == 64**3
    box.surrounding_nodes()
    assert box.num_pts() == 65**3


def test_box_surrounding_nodes_dir():
    """surrounding_nodes(0) makes only x direction nodal."""
    box = blockamr.Box([0, 0, 0], [63, 63, 63])
    box.surrounding_nodes(0)
    assert box.num_pts() == 65 * 64 * 64


def test_box_enclosed_cells():
    """enclosed_cells() converts nodal box back to cell-centred."""
    NODE = blockamr.IndexType.CellIndex.NODE
    nodal = blockamr.IndexType(NODE, NODE, NODE)
    box = blockamr.Box([0, 0, 0], [64, 64, 64], nodal)
    assert box.num_pts() == 65**3
    box.enclosed_cells()
    assert box.num_pts() == 64**3
    assert box.cell_centered()


def test_box_enclosed_cells_dir():
    """enclosed_cells(0) converts only x back to cell-centred."""
    NODE = blockamr.IndexType.CellIndex.NODE
    nodal = blockamr.IndexType(NODE, NODE, NODE)
    box = blockamr.Box([0, 0, 0], [64, 64, 64], nodal)
    box.enclosed_cells(0)
    assert box.num_pts() == 64 * 65 * 65


# --- BoxArray centering tests ---


def test_boxarray_surrounding_nodes():
    """BoxArray.surrounding_nodes(0) makes x-face BoxArray."""
    box = blockamr.Box([0, 0, 0], [63, 63, 63])
    ba = blockamr.BoxArray(box)
    ba.max_size(32)
    ba.surrounding_nodes(0)
    NODE = blockamr.IndexType.CellIndex.NODE
    CELL = blockamr.IndexType.CellIndex.CELL
    expected = blockamr.IndexType(NODE, CELL, CELL)
    assert ba.ix_type() == expected


def test_boxarray_convert():
    """BoxArray.convert() changes centering."""
    NODE = blockamr.IndexType.CellIndex.NODE
    nodal = blockamr.IndexType(NODE, NODE, NODE)
    box = blockamr.Box([0, 0, 0], [63, 63, 63])
    ba = blockamr.BoxArray(box)
    ba.max_size(32)
    ba.convert(nodal)
    assert ba.ix_type() == nodal


def test_boxarray_index_type_kwarg():
    """BoxArray with index_type kwarg applies centering at construction."""
    NODE = blockamr.IndexType.CellIndex.NODE
    CELL = blockamr.IndexType.CellIndex.CELL
    xface = blockamr.IndexType(NODE, CELL, CELL)
    box = blockamr.Box([0, 0, 0], [63, 63, 63])
    ba = blockamr.BoxArray(box, index_type=xface)
    ba.max_size(32)
    assert ba.ix_type() == xface


def test_face_centred_multifab_shape():
    """Face-centred MultiFab patches have correct shape (33x32x32 for x-face)."""
    import numpy as np

    box = blockamr.Box([0, 0, 0], [63, 63, 63])
    ba = blockamr.BoxArray(box)
    ba.max_size(32)
    ba.surrounding_nodes(0)
    dm = blockamr.DistributionMapping(ba)
    mf = blockamr.MultiFab(ba, dm, 1, 0)
    for mfi in blockamr.MFIterator(mf):
        arr = mf.array(mfi)
        shape = arr.shape
        # x-face patches: 33 in x (nodal), 32 in y,z (cell)
        assert shape[0] == 33
        assert shape[1] == 32
        assert shape[2] == 32
        assert shape[3] == 1
        break


def test_face_centred_multifab_roundtrip():
    """Write and read back data from a face-centred MultiFab."""
    import numpy as np

    box = blockamr.Box([0, 0, 0], [63, 63, 63])
    ba = blockamr.BoxArray(box)
    ba.max_size(32)
    ba.surrounding_nodes(0)
    dm = blockamr.DistributionMapping(ba)
    mf = blockamr.MultiFab(ba, dm, 1, 0)
    for mfi in blockamr.MFIterator(mf):
        arr = mf.array(mfi)
        arr[:, :, :, 0] = 42.0
    for mfi in blockamr.MFIterator(mf):
        arr = mf.array(mfi)
        assert np.allclose(arr[:, :, :, 0], 42.0)
        break
