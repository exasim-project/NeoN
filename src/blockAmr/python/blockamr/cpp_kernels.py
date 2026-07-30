# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""C++ (AMReX ``ParallelFor``) explicit-kernel wrappers — peer of :mod:`cell_kernels_3d`.

A scheme's ``build_cpp_kernel()`` returns one of these, mirroring how its
``build_spatial_kernel()`` returns a jax functor. Each wrapper binds one accumulate
kernel, adds ``coeff * op(phi)`` into the scratch source MultiFab, and pulls its own
arguments off the term. The finishing ``euler_update`` axpy lives in ``CppBackend``.
"""


def _bindings():
    # Lazy: schemes import this at load time, before ``blockamr`` finishes initialising.
    import blockamr

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
