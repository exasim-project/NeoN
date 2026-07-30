// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_Array4.H>
#include <AMReX_GpuQualifiers.H>

// The 7-point face-coefficient stencil at ONE cell: the negSumDiag diagonal
// alpha - (aE+aW+aN+aS+aT+aB) and its off-diagonal partner. Every matrix format, smoother and
// residual in the component must agree BIT-FOR-BIT on this arithmetic
// (matrixFree/faceCoeffOp.hpp:27-33, gmgKokkos/kernels.hpp:15-17), and that holds only while the
// association order exists in exactly one place -- so it lives here rather than in gmg/, whose
// kernels are one of four consumers.

namespace blockamr::la
{

// The 6 face-coefficient VALUES at one cell (a loop body holds Array4 views, not FabArrays).
template<class T>
struct FaceCoeffVals
{
    T aE, aW, aN, aS, aT, aB;
};

// aE/aN/aT are the HIGH faces (upper, index +1), aW/aS/aB the LOW ones (lower, index 0).
// C is the type the CELL computes in; TC the coefficient STORAGE type, narrower on a
// reduced-precision GMG level (gmg/bf16.hpp) -- hence two parameters and not one.
template<class C, class TC>
AMREX_GPU_HOST_DEVICE FaceCoeffVals<C> loadFaceCoeffs(
    const amrex::Array4<const TC>& ux,
    const amrex::Array4<const TC>& lx,
    const amrex::Array4<const TC>& uy,
    const amrex::Array4<const TC>& ly,
    const amrex::Array4<const TC>& uz,
    const amrex::Array4<const TC>& lz,
    int i,
    int j,
    int k
) noexcept
{
    return {
        static_cast<C>(ux(i + 1, j, k)),
        static_cast<C>(lx(i, j, k)),
        static_cast<C>(uy(i, j + 1, k)),
        static_cast<C>(ly(i, j, k)),
        static_cast<C>(uz(i, j, k + 1)),
        static_cast<C>(lz(i, j, k))
    };
}

// Summation order aE+aW+aN+aS+aT+aB must stay bit-for-bit identical at every site.
// `alpha` keeps its own deduced type and the result its natural one: where the diagonal SOURCE
// is stored narrower or wider than the cell computes, the single rounding must happen at the
// caller's own `const C diag = ...`, exactly as it does when the formula is written inline.
template<class A, class C>
AMREX_GPU_HOST_DEVICE auto stencilDiag(A alpha, const FaceCoeffVals<C>& c) noexcept
{
    return alpha - (c.aE + c.aW + c.aN + c.aS + c.aT + c.aB);
}

// Same rule for the neighbour values: P is the FIELD's type, which is independent of the
// coefficients' -- a level may store fp64 fields against bf16 coefficients.
template<class C, class P>
AMREX_GPU_HOST_DEVICE auto
stencilOffDiag(const FaceCoeffVals<C>& c, P pE, P pW, P pN, P pS, P pT, P pB) noexcept
{
    return c.aE * pE + c.aW * pW + c.aN * pN + c.aS * pS + c.aT * pT + c.aB * pB;
}

} // namespace blockamr::la
