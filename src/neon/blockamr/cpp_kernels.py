# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""C++ (AMReX ``ParallelFor``) explicit-kernel wrappers.

Peer of :mod:`cell_kernels_3d` (the jax functors). A scheme's
``build_cpp_kernel()`` returns one of these, exactly mirroring how its
``build_spatial_kernel()`` returns a jax functor — so the cpp kernel for a
scheme lives next to the jax one, on the scheme class.

Each wrapper binds one accumulate kernel (``div_*_acc`` / ``laplacian_acc`` /
``grad_acc``), adding ``coeff * op(phi)`` into the scratch source MultiFab, and
knows how to pull its arguments off the term. The generic ``euler_update`` axpy
that finishes the forward-Euler step lives in :class:`~.backends.cpp_backend.CppBackend`.
"""


def _bindings():
    # Lazy: keeps this module import-safe during package init (schemes import it
    # at load time, before ``neon.blockamr`` has finished initialising).
    import neon.blockamr as blockamr

    return blockamr


class CppDivAcc:
    """Accumulate ``coeff * div(phi)`` via a named ``div_*_acc`` binding."""

    def __init__(self, binding):
        self.binding = binding

    def add_to(self, src, op, cell_field, lev, geom):
        faces = op.face_field[lev]
        getattr(_bindings(), self.binding)(
            src,
            cell_field.mf[lev],
            faces[0].mf,
            faces[1].mf,
            faces[2].mf,
            geom,
            op.coeff,
            cell_field.ncomp,
        )


class CppLaplacianAcc:
    """Accumulate ``coeff * gamma * laplacian(phi)`` (constant gamma only)."""

    def add_to(self, src, op, cell_field, lev, geom):
        if not isinstance(op.gamma, (int, float)):
            raise NotImplementedError(
                "cpp backend: variable/callable gamma on term 'Laplacian' "
                f"(scheme {op.scheme.type!r}) not supported (parity with jax)"
            )
        _bindings().laplacian_acc(
            src, cell_field.mf[lev], geom, op.coeff * float(op.gamma), cell_field.ncomp
        )


class CppGradAcc:
    """Accumulate ``coeff * grad(phi)`` (scalar phi -> 3-component vector)."""

    def add_to(self, src, op, cell_field, lev, geom):
        _bindings().grad_acc(src, cell_field.mf[lev], geom, op.coeff)
