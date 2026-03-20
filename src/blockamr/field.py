# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import blockamr


class PatchData:
    """Data for a single box/patch during iteration."""

    __slots__ = ("valid_arr", "grown_arr", "box", "geom", "ngrow")

    def __init__(self, valid_arr, grown_arr, box, geom, ngrow):
        self.valid_arr = valid_arr
        self.grown_arr = grown_arr
        self.box = box
        self.geom = geom
        self.ngrow = ngrow


class Field:
    """Wraps MultiFab + Geometry for DSL operators."""

    def __init__(self, mf, geom, name="", box=None, dm=None, max_size=32):
        self.mf = mf
        self.geom = geom
        self.name = name
        self.box = box
        self.dm = dm
        self.max_size = max_size

    @property
    def dx(self):
        return self.geom.cell_size()

    def fill_boundary(self):
        self.mf.fill_boundary(self.geom)

    def patches(self):
        """Yield PatchData for each local box."""
        for mfi in blockamr.MFIterator(self.mf):
            yield PatchData(
                valid_arr=self.mf.host_array(mfi),
                grown_arr=self.mf.array(mfi),
                box=mfi.valid_box(),
                geom=self.geom,
                ngrow=self.mf.n_grow(),
            )


class CellField(Field):
    """Cell-centred field. Constructs MultiFab from domain box."""

    def __init__(self, box, dm, geom, ncomp=1, ngrow=0, max_size=32, name="", memory="default"):
        ba = blockamr.BoxArray(box)
        ba.max_size(max_size)
        mf = blockamr.MultiFab(ba, dm, ncomp, ngrow, memory=memory)
        super().__init__(mf, geom, name=name, box=box, dm=dm, max_size=max_size)


class NodalField(Field):
    """Node-centred field (nodal in all directions)."""

    def __init__(self, box, dm, geom, ncomp=1, ngrow=0, max_size=32, name="", memory="default"):
        ba = blockamr.BoxArray(box)
        ba.max_size(max_size)
        ba.surrounding_nodes()
        mf = blockamr.MultiFab(ba, dm, ncomp, ngrow, memory=memory)
        super().__init__(mf, geom, name=name, box=box, dm=dm)


class FaceField:
    """Face-centred field. Holds 3 Fields, one per spatial direction."""

    def __init__(self, box, dm, geom, ncomp=1, ngrow=0, max_size=32, name="", memory="default"):
        self.geom = geom
        self.name = name
        self._components = []
        for d in range(3):
            ba = blockamr.BoxArray(box)
            ba.max_size(max_size)
            ba.surrounding_nodes(d)
            mf = blockamr.MultiFab(ba, dm, ncomp, ngrow, memory=memory)
            direction_name = ["x", "y", "z"][d]
            self._components.append(
                Field(mf, geom, name=f"{name}_{direction_name}")
            )

    @property
    def x(self):
        return self._components[0]

    @property
    def y(self):
        return self._components[1]

    @property
    def z(self):
        return self._components[2]

    def __getitem__(self, d):
        return self._components[d]

    def fill_boundary(self):
        for comp in self._components:
            comp.fill_boundary()
