# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import math

import neon.blockamr as blockamr


def _padded_capacity(required, current):
    """Compute padded buffer size with hysteresis.

    Pads by 20% on first allocation or when resizing.
    Keeps the current capacity if the required size is within
    the band [60% of capacity, capacity].  Shrinks only when
    the required size drops below 60% (40% waste).
    """
    if current > 0 and required <= current and required >= int(current * 0.6):
        return current
    return math.ceil(required * 1.2)


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
        self._layout = [None] * (mesh.max_level + 1)
        self._padded_cap = [0] * (mesh.max_level + 1)
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

    def build_layout(self, lev, bf=8):
        """Build TileLayout for this level. Call after regrid or first use."""
        self._layout[lev] = blockamr.build_tile_layout(self.mf[lev], bf)

    def layout(self, lev):
        """Return cached TileLayout. Builds on first call."""
        if self._layout[lev] is None:
            self.build_layout(lev)
        return self._layout[lev]

    def contiguous(self, lev):
        """View of MultiFab contiguous buffer (padded if capacity set)."""
        return self.mf[lev].contiguous_array(self._padded_cap[lev])

    def write_back(self, lev, flat_array):
        """Copy flat array into MultiFab (copies only valid portion)."""
        self.mf[lev].copy_from_flat(flat_array)

    def _make_padded_mf(self, lev, ba, dm):
        """Create a MultiFab with hysteresis-padded contiguous buffer."""
        required = blockamr.MultiFab.required_buffer_size(
            ba, dm, self.ncomp, self.ngrow)
        cap = _padded_capacity(required, self._padded_cap[lev])
        self._padded_cap[lev] = cap
        mf = blockamr.MultiFab(
            ba, dm, self.ncomp, self.ngrow,
            memory=self._memory, padded_n_elems=cap)
        mf.set_val(0.0)
        return mf

    def _on_new_level(self, lev, ba, dm):
        self.mf[lev] = self._make_padded_mf(lev, ba, dm)
        self._layout[lev] = None

    def _on_new_level_from_coarse(self, lev, time, ba, dm):
        new_mf = self._make_padded_mf(lev, ba, dm)
        self._layout[lev] = None
        if self._fill_patch:
            self._fill_patch(self.mesh, self, lev, time, target_mf=new_mf)
        self.mf[lev] = new_mf

    def _on_remake_level(self, lev, time, ba, dm):
        new_mf = self._make_padded_mf(lev, ba, dm)
        if self._fill_patch:
            self._fill_patch(self.mesh, self, lev, time, target_mf=new_mf)
        self.mf[lev] = new_mf
        self._layout[lev] = None

    def _on_clear_level(self, lev):
        self.mf[lev] = None
        self._layout[lev] = None
        self._padded_cap[lev] = 0


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
        self._layouts = [None] * (mesh.max_level + 1)  # per level: (fx_layout, fy_layout, fz_layout)
        mesh.register_field(self)

    def __getitem__(self, lev):
        return self._levels[lev]

    def fill_boundary(self, lev):
        self._levels[lev].fill_boundary()

    def build_layout(self, lev, bf=8):
        """Build TileLayout for each face direction at this level."""
        self._layouts[lev] = tuple(
            blockamr.build_tile_layout(self._levels[lev][d].mf, bf)
            for d in range(3))

    def layout(self, lev, d):
        """Return cached TileLayout for direction d. Builds on first call."""
        if self._layouts[lev] is None:
            self.build_layout(lev)
        return self._layouts[lev][d]

    def contiguous(self, lev, d):
        """Zero-copy view of face-d contiguous buffer."""
        return self._levels[lev][d].mf.contiguous_array()

    def _on_new_level(self, lev, ba, dm):
        self._levels[lev] = _FaceFieldLevel(
            ba, dm, self.mesh.geom(lev),
            self.ncomp, self.ngrow, self.name, self._memory,
        )
        self._layouts[lev] = None

    def _on_new_level_from_coarse(self, lev, time, ba, dm):
        self._on_new_level(lev, ba, dm)

    def _on_remake_level(self, lev, time, ba, dm):
        self._on_new_level(lev, ba, dm)

    def _on_clear_level(self, lev):
        self._levels[lev] = None
        self._layouts[lev] = None
