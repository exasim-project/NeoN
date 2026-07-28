# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Source schemes — the explicit (Su) source term's cell-level kernels.

One scheme, ``PointwiseSource``: the term's operand field *is* the coefficient,
so its "discretisation" is the identity read of that field at the cell. It owns
both kernels the way every other scheme does — ``build_spatial_kernel()``
returns the jax functor, ``build_cpp_kernel()`` the ``source_acc`` wrapper — so
the two backends stay two launches of one arithmetic.

Not in ``SCHEME_REGISTRY``: that table is the name-resolved ``fvSchemes`` axis
(``laplacian``/``div``/``grad``), and a source term has no discretisation to
choose. ``ExplicitSource`` carries this object from construction
(``_scheme_explicit = True``), so the dict lookup is never reached.
"""
from __future__ import annotations

from typing import Annotated, ClassVar, Literal, Union

from pydantic import BaseModel, ConfigDict, Discriminator

from ..cell_kernels_3d import Source3D
from ..cpp_kernels import CppSourceAcc


class PointwiseSource(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal["PointwiseSource"] = "PointwiseSource"
    #: The term reads its own cell and nothing else, so it needs no ghost cell
    #: and contributes nothing to the equation's band width or ``required_ngrow``
    #: (both are a ``max`` over the terms).
    stencil_width: int = 0
    #: Declared, never defaulted — the boundary resolver refuses a scheme that
    #: leaves it open (``plans/IBM/design.md`` §4). A pointwise stencil is inside
    #: every shape; ``cross`` is the one the band is defined against.
    stencil_shape: ClassVar[str] = "cross"

    def build_spatial_kernel(self, buf, offsets, strides, coeff=1.0):
        """Return the 3D functor kernel for ``parallel_for`` dispatch.

        Parameters
        ----------
        buf : jax.Array
            The source field's contiguous flat buffer.
        offsets : jax.Array
            Per-box offset of the first *valid* cell, ``(n_boxes_padded,)`` int32.
        strides : jax.Array
            Per-box ``(1, Nx, Nx*Ny)``, ``(n_boxes_padded, 3)`` int32.
        coeff : float
            The term's scalar coefficient.
        """
        return Source3D(buf=buf, offsets=offsets, strides=strides, coeff=coeff)

    def build_cpp_kernel(self):
        return CppSourceAcc()


SourceScheme = Annotated[
    Union[PointwiseSource],
    Discriminator("type"),
]
