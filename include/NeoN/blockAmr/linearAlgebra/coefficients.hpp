// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ginkgo/ginkgo.hpp>

#include <concepts>
#include <cstddef>
#include <memory>
#include <optional>

#include "NeoN/blockAmr/core/fieldLevel.hpp"
#include "NeoN/blockAmr/core/meshLevel.hpp"
#include "NeoN/blockAmr/linearAlgebra/solverConfig.hpp"
#include "NeoN/core/executor/executor.hpp"

namespace blockamr::la
{

// The coefficient handles are deliberately NARROW: a CellFieldLevel is one
// cell-centred amrex::MultiFab, a FaceFieldLevel is the three direction fields
// (core/fieldLevel.hpp).
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
// a MultiFab -- and then the erasure goes HERE, in the handles, not in Matrix.
//
// ABSENCE HAS EXACTLY ONE SPELLING, and it is std::optional on the ONE member that
// varies. There is no empty() sentinel any more: diag, upper and rhs are non-nullable
// by construction, so a format cannot express handing back a missing one and the
// operators no longer check for it.

enum class Symmetry
{
    symmetric,
    asymmetric
};

// The matrix-side coefficient handles, grouped by role -- lduMatrix's split.
//   mesh  -- the layout those handles are over: ba, dm, geom (core/meshLevel.hpp)
//   diag  -- cell-centred, one value per cell. STILL the diagonal SOURCE alpha,
//            not the matrix diagonal (faceCoeffMatrix.hpp).
//   upper -- face-centred, owner-row -> neighbour coupling on the HIGH face
//   lower -- face-centred, neighbour-row -> owner coupling on the LOW face;
//            NULLOPT when the matrix is symmetric
//
// `lower` being nullopt is the INTERFACE saying "there is no low side to write".
// It is NOT a statement about storage: a symmetric format genuinely stores lower[d]
// ALIASED to upper[d] (faceCoeffMatrix.hpp) and merely declines to hand it out,
// because an operator writing both would double every coefficient. The storage
// reading has its own accessor and its own type (FaceCoeffFields::storedLower(),
// a plain FaceFieldLevel), so the two cannot be confused at a call site.
//
// `mesh` is FIRST, and it is here rather than beside these handles at a call site,
// because the coefficients are not self-describing without it: every operator that
// writes a face coefficient needs dx to scale it and the periodicity to fill the
// halo the face average reads across. Passing it separately made a mismatch
// representable -- ops::Laplacian used to carry its own amrex::Geometry, handed in
// at construction, and nothing checked it against the matrix's. It cannot be
// spelled now: the operator reads the mesh off the coefficients it was handed.
struct MatrixCoefficients
{
    MeshLevel mesh;
    CellFieldLevel diag;
    FaceFieldLevel upper;
    std::optional<FaceFieldLevel> lower; // nullopt when symmetric

    bool symmetric() const { return !lower.has_value(); }
};

class LinearSystem;

// What an OPERATOR writes through: the matrix coefficients plus the rhs, which lives on
// the system rather than the matrix. Format-agnostic -- the same handles regardless of
// what is underneath. Only LinearSystem can build one (private ctor + friend).
class Coefficients
{
public:

    // The layout every handle below is over, carried down from the matrix with
    // them. An operator needs dx and the periodicity to write a face coefficient
    // at all, so a Coefficients without this is not sufficient to assemble from --
    // which is the whole claim this type makes.
    MeshLevel mesh;
    CellFieldLevel diag;
    FaceFieldLevel upper;
    std::optional<FaceFieldLevel> lower; // nullopt when symmetric
    CellFieldLevel rhs;
    // Where the operator's kernels launch (blockamr::parallelFor). It comes from
    // the MATRIX -- every format carries one (IsMatrix::executor()) and
    // LinearSystem hands it down here -- rather than being given to the operator
    // at construction: an operator writes the matrix's coefficient FIELDS, so it
    // has to launch where those fields live, and only the matrix knows that.
    NeoN::Executor exec {NeoN::SerialExecutor {}};

    bool symmetric() const { return !lower.has_value(); }

private:

    friend class LinearSystem;

    Coefficients(MatrixCoefficients mc, CellFieldLevel rhs, const NeoN::Executor& exec)
        : mesh(mc.mesh), diag(mc.diag), upper(mc.upper), lower(mc.lower), rhs(rhs), exec(exec)
    {}
};

// What it takes to BE a matrix format. No base class to derive from -- satisfy this and
// you are one.
//
// makePrecond IS THE FORMAT'S JOB, and that is the whole point of it being here rather
// than on the solver. A geometric-multigrid hierarchy is built by rediscretising the
// COEFFICIENTS on a sequence of coarser levels; a solver holds a gko::LinOp, by which
// point the erasure has thrown the coefficients away. So the only object that still can
// build one is the format that owns them. `config` is the SolverConfig the solve was
// asked for -- reused rather than re-spelled as a narrower precond-only struct, because
// it already carries precondKind, the GmgConfig V-cycle knobs, precondCycles and
// precondMlmg, and a second spelling of those is exactly the duplication this refactor
// removes.
//
// Returning NULL means the format DECLINES: it has no way to build that preconditioner
// and says so by handing back nothing, leaving the caller to raise. It is not an error
// signal a format invents a message for -- only the caller knows what it was going to
// do with it. A format still THROWS for a combination it must refuse on its own terms
// (an unbuildable GMG variant, say), and precond="none" returns null without either
// party treating it as a refusal.
//
// name() exists for that message and for nothing else: an erased Matrix cannot
// otherwise say what it is holding, and "declines precond 'gmg'" is not actionable
// without it.
template<typename T>
concept IsMatrix = requires(T t, const T ct, const SolverConfig& config) {
    {
        ct.op()
    } -> std::same_as<std::shared_ptr<const gko::LinOp>>;
    {
        ct.isAssembled()
    } -> std::same_as<bool>;
    {
        t.coefficients()
    } -> std::same_as<MatrixCoefficients>;
    {
        t.zero()
    } -> std::same_as<void>;
    {
        ct.symmetry()
    } -> std::same_as<Symmetry>;
    {
        ct.localRows()
    } -> std::same_as<std::size_t>;
    {
        ct.executor()
    } -> std::same_as<const NeoN::Executor&>;
    {
        ct.makePrecond(config)
    } -> std::same_as<std::shared_ptr<const gko::LinOp>>;
    {
        ct.name()
    } -> std::same_as<const char*>;
};

// What it takes to BE an operator. Declared HERE, beside the coefficients it writes, so
// that linearAlgebra/ never depends on the operators -- the dependency runs ops -> la.
template<typename T>
concept IsOperator = requires(const T t, Coefficients c) {
    {
        t.assemble(c)
    } -> std::same_as<void>;
};

} // namespace blockamr::la
