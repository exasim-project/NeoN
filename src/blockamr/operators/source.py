# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""The two source terms, Sp and Su (cf. NeoN's C++ ``dsl::exp::source``).

:class:`Source` is the **implicit (Sp)** form ``coeff_func(x,y,z,t) * phi``: a
callable coefficient times the equation's solved field. It carries no scheme, so
the cpp backend raises for it (pinned by a green test in
``test_backend_dispatch.py``) and it runs on jax alone.

:class:`ExplicitSource` is the **explicit (Su)** form: one ``CellField``
operand, which *is* the coefficient. It is schemed (``PointwiseSource``) and
therefore runs on both backends, and — being a spatial term — it also emits
band rows under an IBM method (``source x ghostCell``).
"""

from typing import NamedTuple

import jax.numpy as jnp

from ..dsl.eqterm import EqTerm
from ..schemes.source_schemes import PointwiseSource


class SourceKernel(NamedTuple):
    S: object  # Array
    ng: int
    coeff: float

    def __call__(self, phi):
        phi_valid = (
            phi[self.ng : -self.ng, self.ng : -self.ng, self.ng : -self.ng] if self.ng > 0 else phi
        )
        return self.coeff * self.S * phi_valid


class Source(EqTerm):
    """Implicit (Sp) pointwise source term: S(x,y,z,t) * phi.

    coeff_func(x, y, z, t) -> scalar_array evaluated at cell centers.

    Deliberately unschemed: ``scheme`` stays ``None``, so the cpp backend
    raises naming this term rather than launching a kernel that does not exist.
    """

    kind = "spatial"
    scheme_key = "source"

    def __init__(self, coeff_func, field, coeff=1.0):
        super().__init__(field, coeff=coeff, coefficient=coeff_func)
        self.coeff_func = coeff_func

    def build_kernel(self, mfi, t, lev=0):
        """Return a SourceKernel functor for this mfi."""
        ng = self.field.mf[lev].n_grow()
        geom = self.field.mesh.geom(lev)
        lo = mfi.valid_box().small_end()
        dx = geom.cell_size()
        prob_lo = geom.prob_lo()
        bx = mfi.valid_box()
        lo_v = bx.small_end()
        hi_v = bx.big_end()
        nx = hi_v[0] - lo_v[0] + 1
        ny = hi_v[1] - lo_v[1] + 1
        nz = hi_v[2] - lo_v[2] + 1

        xcc = jnp.array([prob_lo[0] + (lo[0] + i + 0.5) * dx[0] for i in range(nx)])
        ycc = jnp.array([prob_lo[1] + (lo[1] + j + 0.5) * dx[1] for j in range(ny)])
        zcc = jnp.array([prob_lo[2] + (lo[2] + k + 0.5) * dx[2] for k in range(nz)])

        X, Y, Z = jnp.meshgrid(xcc, ycc, zcc, indexing="ij")
        S = self.coeff_func(X, Y, Z, t)
        return SourceKernel(S=S, ng=ng, coeff=self.coeff)


class ExplicitSource(EqTerm):
    """Explicit (Su) source term: ``coeff * S``, with ``S`` a CellField.

    The operand *is* the coefficient — ``exp.source(S)``, the one-argument
    overload of NeoN's C++ ``dsl::exp::source`` — so there is nothing to
    evaluate per cell beyond reading ``S`` there. That makes it the only
    spatial term whose operand is not the equation's solved field.

    Schemed from construction (``_scheme_explicit = True``), so
    ``dsl.solve._resolve_schemes`` never looks ``"source"`` up in the
    ``fvSchemes`` dict: there is no discretisation to choose.
    """

    kind = "spatial"
    scheme_key = "source"
    _scheme_operator = "source"

    def __init__(self, field, coeff=1.0, scheme=None):
        super().__init__(field, coeff=coeff, scheme=scheme or PointwiseSource())
        self._scheme_explicit = True

    def build_kernel_3d(self, ctx, t):
        """Build the 3D spatial kernel from TiledContext.

        The source field's flat buffer with per-box *valid-start* offsets and
        strides — the same gather ``Div.build_kernel_3d`` does for its face
        buffers, and read inside the Pallas kernel by the tile's ``box_id``.
        """
        if self.field.ncomp != 1:
            raise NotImplementedError(
                f"the jax backend needs a one-component explicit source, but "
                f"'{self.field.name}' has ncomp = {self.field.ncomp}: parallel_for runs "
                "the kernel once per component by shifting the *phi* tiles, and the "
                "source buffer is not shifted with them — every component would read "
                "component 0. The cpp backend accumulates a vector source natively."
            )

        lev = ctx.lev
        mf = self.field.mf[lev]
        ng = mf.n_grow()
        meta = mf.fab_metadata()
        n_boxes = len(meta)

        offsets = []
        strides = []
        for b in range(n_boxes):
            off, Nx, Ny = int(meta[b][0]), int(meta[b][1]), int(meta[b][2])
            offsets.append(off + ng + Nx * ng + Nx * Ny * ng)
            strides.append([1, Nx, Nx * Ny])

        # Padded to the next power of two, the way parallel_for pads its own
        # per-box arrays: the padding entries are indexed by dead tiles only.
        mb = 1
        while mb < n_boxes:
            mb <<= 1
        while len(offsets) < mb:
            offsets.append(offsets[0])
            strides.append(strides[0])

        return self.scheme.build_spatial_kernel(
            buf=mf.contiguous_array(),
            offsets=jnp.array(offsets, dtype=jnp.int32),
            strides=jnp.array(strides, dtype=jnp.int32),
            coeff=self.coeff,
        )
