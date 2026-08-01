// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

#include "NeoN/blockAmr/core/bc.hpp"
#include "NeoN/blockAmr/core/fieldLevel.hpp"
#include "NeoN/blockAmr/core/meshLevel.hpp"
#include "NeoN/blockAmr/linearAlgebra/transfer.hpp"
#include "NeoN/core/executor/executor.hpp"

// The matrix. AMReX and NeoN only -- deliberately NO Ginkgo, so an operator writing face
// coefficients (operators/laplacian.hpp) never compiles against one; the LinOp and the GMG
// hierarchy are built from it by the free functions in linearAlgebra/ginkgo/adapt.hpp.
// It -- and ONLY it -- applies the homogeneous domain BC; the operator's half of that
// contract: operators/laplacian.hpp.

namespace blockamr::la
{

// Whether the format allocates a separate LOW side. Lives here, with the only storage that
// answers to it.
enum class Symmetry
{
    symmetric,
    asymmetric
};

namespace detail
{

/* @brief Allocate the coefficient fields `mesh` implies. A free function rather than a base
 *        class: the format holds the fields itself, and this is only the derivation of the
 *        AMReX layouts from the mesh.
 */
inline void allocateCoefficients(
    const MeshLevel& mesh,
    Symmetry sym,
    CellFieldLevel& alpha,
    FaceFieldLevel& upper,
    std::optional<FaceFieldLevel>& lower
)
{
    // MultiFabs are not zero-initialised (the arena recycles memory), so a freshly built
    // matrix is explicitly zeroed -- callers write only what their operator contributes.
    auto zeroed = [](const amrex::BoxArray& ba, const amrex::DistributionMapping& dm)
    {
        auto mf = std::make_shared<amrex::MultiFab>(ba, dm, 1, 0);
        mf->setVal(0.0);
        return mf;
    };
    const bool asym = (sym == Symmetry::asymmetric);
    alpha = CellFieldLevel {zeroed(mesh.ba, mesh.dm)};
    FaceFieldLevel low {};
    for (int d = 0; d < 3; ++d)
    {
        const auto i = static_cast<std::size_t>(d);
        const amrex::BoxArray fba = amrex::convert(mesh.ba, amrex::IntVect::TheDimensionVector(d));
        upper.dir[i] = zeroed(fba, mesh.dm);
        if (asym)
        {
            low.dir[i] = zeroed(fba, mesh.dm);
        }
    }
    if (asym)
    {
        lower = low;
    }
}

inline void
zeroCoefficients(CellFieldLevel& alpha, FaceFieldLevel& upper, std::optional<FaceFieldLevel>& lower)
{
    (*alpha).setVal(0.0);
    for (int d = 0; d < 3; ++d)
    {
        upper[d].setVal(0.0);
        // Nothing separate to zero when symmetric.
        if (lower.has_value())
        {
            (*lower)[d].setVal(0.0);
        }
    }
}

} // namespace detail

/* @class MFFaceCoeffs
 * @brief Matrix-free face-coefficient storage: no MATRIX is ever assembled -- la::toLinOp
 *        turns these fields into a FaceCoeffOp per solve. Build with symmetric()/
 *        asymmetric(), then write the coefficient fields directly.
 */
class MFFaceCoeffs
{
public:

    // The fields below sit behind shared_ptr (MultiFab is not copyable), so a copy of the
    // format SHARES them and a write through one is seen through the other.
    NeoN::Executor exec {NeoN::SerialExecutor {}};
    la::BcArray bc {};
    // The layout the fields were ALLOCATED from, and nothing repoints them, so it is the one
    // source for ba/dm rather than a second one competing with alpha's.
    MeshLevel mesh;
    // The negSumDiag seam: `alpha` is the cell-centred diagonal SOURCE (ddt/Sp/reaction), NOT
    // the matrix diagonal alpha - (aE+aW+aN+aS+aT+aB), which an operator's += must not land on;
    // deriving that one is the stencil's job.
    CellFieldLevel alpha;
    FaceFieldLevel upper;
    // No low side to WRITE when symmetric: an operator writing both would double every
    // coefficient. Storage still has one -- storedLower().
    std::optional<FaceFieldLevel> lower; // nullopt when symmetric

    // `bc` defaults to all-periodic (BcArray 0 == periodic).
    static MFFaceCoeffs
    symmetric(const NeoN::Executor& exec, MeshLevel mesh, const la::BcArray& bc = {})
    {
        return MFFaceCoeffs(exec, std::move(mesh), Symmetry::symmetric, bc);
    }

    static MFFaceCoeffs
    asymmetric(const NeoN::Executor& exec, MeshLevel mesh, const la::BcArray& bc = {})
    {
        return MFFaceCoeffs(exec, std::move(mesh), Symmetry::asymmetric, bc);
    }

    void zero() { detail::zeroCoefficients(alpha, upper, lower); }

    /* @brief The STORED low side, which always exists -- `upper` itself when symmetric, the
     *        alias FaceCoeffOp's convention wants. Deliberately NOT `lower`'s reading, where a
     *        symmetric matrix has none; the differing types keep the two from being confused.
     */
    FaceFieldLevel storedLower() const { return lower.value_or(upper); }

    // Derived from `lower`, so no stored enum can drift from the storage.
    Symmetry symmetry() const { return lower ? Symmetry::asymmetric : Symmetry::symmetric; }

    // Rows this rank owns. localCount(), NOT numPts(): numPts() counts EVERY rank's cells.
    std::size_t localRows() const { return la::localCount(*alpha); }

private:

    MFFaceCoeffs(
        const NeoN::Executor& executor, MeshLevel meshLevel, Symmetry sym, const la::BcArray& bcs
    )
        : exec(executor), bc(bcs), mesh(std::move(meshLevel))
    {
        detail::allocateCoefficients(mesh, sym, alpha, upper, lower);
    }
};

// Absence has exactly ONE spelling and only `lower` has it: with CellFieldLevel/FaceFieldLevel
// non-nullable, "empty" is not expressible, which is why _la_matrix_probe's diag_empty/
// upper_empty keys emit a literal false. Turn either into an optional and this fires.
// (Was asserted in linearAlgebra/coefficientsConcepts.cpp, deleted with the concepts.)
static_assert(std::is_same_v<decltype(MFFaceCoeffs::alpha), CellFieldLevel>);
static_assert(std::is_same_v<decltype(MFFaceCoeffs::upper), FaceFieldLevel>);
static_assert(std::is_same_v<decltype(MFFaceCoeffs::lower), std::optional<FaceFieldLevel>>);
// The mesh TRAVELS with the coefficients, not optional and not a pointer: an operator cannot
// write a face coefficient without dx, and passing it alongside made a mismatch representable
// (ops::Laplacian used to hold its own amrex::Geometry).
static_assert(std::is_same_v<decltype(MFFaceCoeffs::mesh), MeshLevel>);

} // namespace blockamr::la
