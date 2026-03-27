# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import neon.blockamr as blockamr


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
                valid_arr=self.mf.copy_to_host(mfi),
                grown_arr=self.mf.array(mfi),
                box=mfi.valid_box(),
                geom=self.geom,
                ngrow=self.mf.n_grow(),
            )


class CellField:
    """Cell-centred field. Works with both Mesh and AmrMesh."""

    def __init__(self, mesh, ncomp=1, ngrow=0, name="", fill_patch=None, memory="default"):
        self.mesh = mesh
        self.name = name
        self.ncomp = ncomp
        self.ngrow = ngrow
        self._fill_patch = fill_patch
        self._memory = memory
        self.mf = [None] * (mesh.max_level + 1)
        mesh.register_field(self)

    def __getitem__(self, lev):
        return Field(self.mf[lev], self.mesh.geom(lev), name=self.name)

    def fill_patch(self, lev, time):
        if self.mf[lev] is None:
            return
        if self._fill_patch:
            self._fill_patch(self.mesh, self, lev, time)
        else:
            self.mf[lev].fill_boundary(self.mesh.geom(lev))

    def _on_new_level(self, lev, ba, dm):
        self.mf[lev] = blockamr.MultiFab(ba, dm, self.ncomp, self.ngrow, memory=self._memory)
        self.mf[lev].set_val(0.0)

    def _on_new_level_from_coarse(self, lev, time, ba, dm):
        self.mf[lev] = blockamr.MultiFab(ba, dm, self.ncomp, self.ngrow, memory=self._memory)
        self.mf[lev].set_val(0.0)
        if self._fill_patch:
            self._fill_patch(self.mesh, self, lev, time)

    def _on_remake_level(self, lev, time, ba, dm):
        new_mf = blockamr.MultiFab(ba, dm, self.ncomp, self.ngrow, memory=self._memory)
        if self._fill_patch:
            self._fill_patch(self.mesh, self, lev, time, target_mf=new_mf)
        self.mf[lev] = new_mf

    def _on_clear_level(self, lev):
        self.mf[lev] = None


class NodalField(Field):
    """Node-centred field (nodal in all directions)."""

    def __init__(self, box, dm, geom, ncomp=1, ngrow=0, max_size=32, name="", memory="default"):
        ba = blockamr.BoxArray(box)
        ba.max_size(max_size)
        ba.surrounding_nodes()
        mf = blockamr.MultiFab(ba, dm, ncomp, ngrow, memory=memory)
        mf.set_val(0.0)
        super().__init__(mf, geom, name=name, box=box, dm=dm)


class _FaceFieldLevel:
    """Per-level face data. 3 Fields, one per direction."""

    def __init__(self, ba, dm, geom, ncomp, ngrow, name, memory="default"):
        self.geom = geom
        self._components = []
        for d in range(3):
            ba.surrounding_nodes(d)
            mf = blockamr.MultiFab(ba, dm, ncomp, ngrow, memory=memory)
            mf.set_val(0.0)
            ba.enclosed_cells(d)
            self._components.append(Field(mf, geom, name=f"{name}_{'xyz'[d]}"))

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
        for c in self._components:
            c.fill_boundary()


class FaceField:
    """Face-centred field. Works with both Mesh and AmrMesh."""

    def __init__(self, mesh, ncomp=1, ngrow=0, name="", memory="default"):
        self.mesh = mesh
        self.name = name
        self.ncomp = ncomp
        self.ngrow = ngrow
        self._memory = memory
        self._levels = [None] * (mesh.max_level + 1)
        mesh.register_field(self)

    def __getitem__(self, lev):
        return self._levels[lev]

    def fill_boundary(self, lev):
        self._levels[lev].fill_boundary()

    def _on_new_level(self, lev, ba, dm):
        self._levels[lev] = _FaceFieldLevel(
            ba, dm, self.mesh.geom(lev),
            self.ncomp, self.ngrow, self.name, self._memory,
        )

    def _on_new_level_from_coarse(self, lev, time, ba, dm):
        self._on_new_level(lev, ba, dm)

    def _on_remake_level(self, lev, time, ba, dm):
        self._on_new_level(lev, ba, dm)

    def _on_clear_level(self, lev):
        self._levels[lev] = None
