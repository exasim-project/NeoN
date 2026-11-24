# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""C++ (AMReX ``ParallelFor``) explicit-kernel wrappers.

A scheme's (or term's) ``build_cpp_kernel()`` returns one of these. Each wrapper binds
one accumulate kernel, adds ``coeff * op(phi)`` into the scratch source MultiFab, and
pulls its own arguments off the term. The finishing ``euler_update`` axpy lives in
:mod:`blockamr.explicit`.
"""

import numpy as np


def _bindings():
    # Lazy: schemes import this at load time, before ``blockamr`` finishes initialising.
    import blockamr

    return blockamr


class CppDivAcc:
    """Accumulate ``coeff * div(phi)`` via a named ``div_*_acc`` binding."""

    def __init__(self, binding):
        self.binding = binding

    def add_to(self, src, op, cell_field, lev, geom, t):
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

    def add_to(self, src, op, cell_field, lev, geom, t):
        _bindings().laplacian_acc(
            src, cell_field.mf[lev], geom, op.coeff * _constant_gamma(op, t), cell_field.ncomp
        )


def _constant_gamma(op, t):
    """``op.gamma`` as a number: a callable is accepted when it is provably constant."""
    gamma = op.gamma
    if isinstance(gamma, (int, float)):
        return float(gamma)
    if callable(gamma):
        # Two asymmetric probes: a constant callable gamma takes the constant path.
        g1 = float(np.asarray(gamma(np.array([0.1]), np.array([0.2]), np.array([0.3]), t))[0])
        g2 = float(np.asarray(gamma(np.array([0.7]), np.array([0.4]), np.array([0.15]), t))[0])
        if abs(g1 - g2) < 1e-14 * (abs(g1) + 1.0):
            return g1
    raise NotImplementedError(
        "variable gamma on term 'Laplacian' "
        f"(scheme {op.scheme.type!r}) is not supported"
    )


class CppGradAcc:
    """Accumulate ``coeff * grad(phi)`` (scalar phi -> 3-component vector)."""

    def add_to(self, src, op, cell_field, lev, geom, t):
        _bindings().grad_acc(src, cell_field.mf[lev], geom, op.coeff)


class CppSourceAcc:
    """Accumulate ``coeff * S(x, y, z, t) * phi``.

    ``S`` is a Python callable, so it is evaluated on the host at the cell centres;
    the product ``S * phi`` is staged into a scratch MultiFab that ``source_acc`` adds.
    """

    def add_to(self, src, op, cell_field, lev, geom, t):
        blockamr = _bindings()
        mf = cell_field.mf[lev]
        ncomp = cell_field.ncomp
        dx = geom.cell_size()
        prob_lo = geom.prob_lo()
        products = []
        for mfi in blockamr.MFIterator(mf):
            bx = mfi.valid_box()
            lo = bx.small_end()
            hi = bx.big_end()
            cc = [prob_lo[d] + (np.arange(lo[d], hi[d] + 1) + 0.5) * dx[d] for d in range(3)]
            X, Y, Z = np.meshgrid(*cc, indexing="ij")
            S = np.broadcast_to(np.asarray(op.coeff_func(X, Y, Z, t), dtype=float), X.shape)
            phi = mf.copy_to_host(mfi)
            products.append(np.asfortranarray(S[..., None] * phi))
        mesh = cell_field.mesh
        tmp = blockamr.MultiFab(
            mesh.box_array(lev), mesh.dm(lev), ncomp, 0, memory=cell_field._memory
        )
        tmp.copy_arrays(products)
        blockamr.source_acc(src, tmp, op.coeff, ncomp)
