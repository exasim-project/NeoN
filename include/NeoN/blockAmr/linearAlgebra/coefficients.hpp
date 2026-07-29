// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_MultiFab.H>

#include <ginkgo/ginkgo.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <memory>

#include "NeoN/core/executor/executor.hpp"

namespace blockamr::la
{

// The coefficient handles are deliberately NARROW: a CellView is one cell-centred
// amrex::MultiFab, a FaceView is the three direction fields.
//
// They stand for BOTH formats, assembled and matrix-free, and neither needs a type
// erasure or a variant inside them. The reason is that assembleFaceCoeffCsr already
// takes seven amrex::MultiFabs and reads them host-side to build its arrays
// (sparse/csr.cpp) -- MultiFabs are the assembled path's input, not a staging buffer
// added for this interface's convenience. So CsrMatrix holds exactly what MFFaceCoeffs
// holds and hands out these same handles (faceCoeffMatrix.hpp); what it needs on top is
// only an assembly-freshness flag. Do not generalise these types speculatively. Revisit
// only for a format whose storage is genuinely not MultiFab-shaped -- an ELL/banded
// matrix owning its own arrays, or a device-assembled CSR that never round-trips through
// a MultiFab -- and then the erasure goes HERE, in the views, not in Matrix.

enum class Symmetry
{
    symmetric,
    asymmetric
};

// Non-owning handle to one cell-centred field. Empty means "not provided".
struct CellView
{
    amrex::MultiFab* ptr = nullptr;

    bool empty() const { return ptr == nullptr; }
};

// Non-owning handle to the three face-centred direction fields. Empty means none of
// the three is provided -- which is how a symmetric matrix reports its absent `lower`.
struct FaceView
{
    std::array<amrex::MultiFab*, 3> dir {nullptr, nullptr, nullptr};

    bool empty() const { return dir[0] == nullptr && dir[1] == nullptr && dir[2] == nullptr; }
};

// The matrix-side coefficient handles, grouped by role -- lduMatrix's split.
//   diag  -- cell-centred, one value per cell
//   upper -- face-centred, owner-row -> neighbour coupling on the HIGH face
//   lower -- face-centred, neighbour-row -> owner coupling on the LOW face;
//            absent when the matrix is symmetric
struct MatrixCoefficients
{
    CellView diag;
    FaceView upper;
    FaceView lower; // empty when symmetric

    bool symmetric() const { return lower.empty(); }
};

class LinearSystem;

// What an OPERATOR writes through: the matrix coefficients plus the rhs, which lives on
// the system rather than the matrix. Format-agnostic -- the same handles regardless of
// what is underneath. Only LinearSystem can build one (private ctor + friend).
class Coefficients
{
public:

    CellView diag;
    FaceView upper;
    FaceView lower; // empty when symmetric
    CellView rhs;
    // Where the operator's kernels launch (blockamr::parallelFor). It comes from
    // the MATRIX -- every format carries one (IsMatrix::executor()) and
    // LinearSystem hands it down here -- rather than being given to the operator
    // at construction: an operator writes the matrix's coefficient FIELDS, so it
    // has to launch where those fields live, and only the matrix knows that.
    NeoN::Executor exec {NeoN::SerialExecutor {}};

    bool symmetric() const { return lower.empty(); }

private:

    friend class LinearSystem;

    Coefficients(MatrixCoefficients mc, CellView rhs, const NeoN::Executor& exec)
        : diag(mc.diag), upper(mc.upper), lower(mc.lower), rhs(rhs), exec(exec)
    {}
};

// What it takes to BE a matrix format. No base class to derive from -- satisfy this and
// you are one.
template<typename T>
concept IsMatrix = requires(T t, const T ct) {
    { ct.op() } -> std::same_as<std::shared_ptr<const gko::LinOp>>;
    { ct.isAssembled() } -> std::same_as<bool>;
    { t.coefficients() } -> std::same_as<MatrixCoefficients>;
    { t.zero() } -> std::same_as<void>;
    { ct.symmetry() } -> std::same_as<Symmetry>;
    { ct.localRows() } -> std::same_as<std::size_t>;
    { ct.executor() } -> std::same_as<const NeoN::Executor&>;
};

// What it takes to BE an operator. Declared HERE, beside the coefficients it writes, so
// that linearAlgebra/ never depends on the operators -- the dependency runs ops -> la.
template<typename T>
concept IsOperator = requires(const T t, Coefficients c) {
    { t.assemble(c) } -> std::same_as<void>;
};

} // namespace blockamr::la
