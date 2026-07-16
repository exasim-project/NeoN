# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""TiledContext: dispatch context for tiled Pallas kernels.

Replaces BoxContext. Constructed once per level by the dispatch layer,
passed to op.build_kernel_3d(ctx, t). Operators read what they need —
Div reads face data, Laplacian reads only dh.
"""


class TiledContext:
    """Dispatch context for operators inside tiled Pallas kernels.

    Constructed by _dispatch_level with face data from FaceFields.
    box_id is set per-tile inside the Pallas kernel via with_box_id().

    Parameters
    ----------
    dh : tuple
        Cell spacing (dx, dy, dz).
    ng : int
        Ghost cell width.
    lev : int
        AMR level index.
    face_refs : tuple of 3, optional
        Pallas refs to face contiguous buffers (fx_ref, fy_ref, fz_ref).
    face_offsets : tuple of 3, optional
        Per-box face offsets: (fxo, fyo, fzo), each (n_boxes_padded,) int32.
    face_strides : tuple of 3, optional
        Per-box face strides: (fxs, fys, fzs), each (n_boxes_padded, 3) int32.
    box_id : traced int32, optional
        Current tile's box index. Set inside the Pallas kernel.
    """

    __slots__ = ('dh', 'ng', 'lev',
                 'face_refs', 'face_offsets', 'face_strides', 'box_id')

    def __init__(self, dh, ng, lev,
                 face_refs=None, face_offsets=None, face_strides=None,
                 box_id=None):
        self.dh = dh
        self.ng = ng
        self.lev = lev
        self.face_refs = face_refs
        self.face_offsets = face_offsets
        self.face_strides = face_strides
        self.box_id = box_id

    def with_box_id(self, box_id):
        """Return a copy with box_id set (used inside Pallas kernel)."""
        return TiledContext(
            dh=self.dh, ng=self.ng, lev=self.lev,
            face_refs=self.face_refs,
            face_offsets=self.face_offsets,
            face_strides=self.face_strides,
            box_id=box_id)

    def with_face_refs(self, face_refs):
        """Return a copy with Pallas face refs bound (inside pallas_call)."""
        return TiledContext(
            dh=self.dh, ng=self.ng, lev=self.lev,
            face_refs=face_refs,
            face_offsets=self.face_offsets,
            face_strides=self.face_strides,
            box_id=self.box_id)
