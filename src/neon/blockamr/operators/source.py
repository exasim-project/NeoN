# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

from typing import NamedTuple

import jax.numpy as jnp

from ..dsl.eqterm import EqTerm


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
    """Pointwise source term: S(x,y,z,t) * phi.

    coeff_func(x, y, z, t) -> scalar_array evaluated at cell centers.
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
