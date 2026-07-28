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
    # at load time, before ``blockamr`` has finished initialising).
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


class CppSourceAcc:
    """Accumulate ``coeff * S`` — the explicit (Su) source, whose operand *is*
    the coefficient.

    The only wrapper that reads a field other than the equation's solved one, so
    it is also the only one that has to check they are laid out alike:
    ``sourceAcc`` iterates ``MFIter(S)`` and indexes the destination by the same
    ``mfi``, which on a different box array is undefined behaviour rather than
    an error.
    """

    def add_to(self, src, op, cell_field, lev, geom):
        source = op.field
        if source.mesh is not cell_field.mesh:
            raise ValueError(
                f"cpp backend: the source field '{source.name}' is on a different mesh "
                f"than '{cell_field.name}'; a source term is accumulated box by box and "
                "the two must share a box array."
            )
        if source.ncomp != cell_field.ncomp:
            raise ValueError(
                f"cpp backend: the source field '{source.name}' has ncomp = {source.ncomp} "
                f"but '{cell_field.name}' has {cell_field.ncomp}; the accumulate writes "
                "component for component."
            )
        _bindings().source_acc(src, source.mf[lev], float(op.coeff), cell_field.ncomp)
