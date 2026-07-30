// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>

#include <ginkgo/ginkgo.hpp>

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <utility>

#include "NeoN/blockAmr/core/bc.hpp"
#include "NeoN/blockAmr/core/fieldLevel.hpp"
#include "NeoN/blockAmr/core/meshLevel.hpp"
#include "NeoN/blockAmr/linearAlgebra/coefficients.hpp"
#include "NeoN/blockAmr/linearAlgebra/krylov/executor.hpp"
#include "NeoN/blockAmr/linearAlgebra/matrixFree/faceCoeffOp.hpp"
#include "NeoN/blockAmr/linearAlgebra/precond.hpp"
#include "NeoN/blockAmr/linearAlgebra/sparse/csr.hpp"
#include "NeoN/blockAmr/linearAlgebra/transfer.hpp"
#include "NeoN/core/executor/executor.hpp"

// The two concrete matrix FORMATS, both satisfying IsMatrix; they differ only in what op()
// hands back. Both -- and ONLY these -- apply the homogeneous domain BC; the operator's half
// of that contract: operators/laplacian.hpp; report/blockamr-linear-algebra-notes.md.

namespace blockamr::la
{

namespace detail
{

/* @brief Allocate the coefficient fields `mesh` implies. Free functions rather than a shared
 *        base: the two formats hold the fields themselves, and the only thing they share is
 *        how the AMReX layouts are derived from the mesh.
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
 * @brief Matrix-free face-coefficient format: op() is a FaceCoeffOp over the coefficient
 *        fields and no MATRIX is ever assembled. One derived field is kept, the diagonal
 *        (see refreshedDiagonal()). Build with symmetric()/asymmetric(), then write the
 *        coefficient fields directly.
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
    // the matrix diagonal alpha - (aE+aW+aN+aS+aT+aB) -- that is `diagonal` below, which an
    // operator's += must not land on.
    CellFieldLevel alpha;
    FaceFieldLevel upper;
    // No low side to WRITE when symmetric: an operator writing both would double every
    // coefficient. Storage still has one -- storedLower().
    std::optional<FaceFieldLevel> lower; // nullopt when symmetric

    // The derived matrix diagonal and its freshness flag. The MultiFab is written IN PLACE, so
    // a copy's own shared_ptr already shares it and only the flag needs one of its own.
    CellFieldLevel diagonal;
    std::shared_ptr<bool> diagonalDirty;

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

    // Built fresh per call, not cached: the operator stages PINNED COPIES of the coefficient
    // fields on the host path, so a cached one would go stale after a write to them on that
    // path. This is a per-solve call, not a per-iteration one.
    std::shared_ptr<const gko::LinOp> op() const
    {
        // PROTOTYPE (C1): no refresh -- the stencil recomputes the centre term inline, so the
        // diagonal passed below is unused.
        return gko::share(la::FaceCoeffOp::create(
            la::makeExecutor(exec),
            exec,
            mesh,
            globalRows(),
            alpha,
            upper,
            // The STORED lower, not the interface's: symmetric ALIASES upper here,
            // exactly FaceCoeffOp's convention; `lower` itself is ABSENT.
            storedLower(),
            // The caller's `bc`: the ghost reflection it drives is what applies the
            // homogeneous domain BC.
            bc,
            nullptr,
            diagonal
        ));
    }

    /* @brief The stored matrix diagonal alpha - (aE+aW+aN+aS+aT+aB): a field of the MATRIX and
     *        not a coefficient, since `alpha` is only its SOURCE. Recomputed lazily off
     *        `diagonalDirty`, which markStale() sets. PROTOTYPE (C1): the stencils bypass it.
     */
    const amrex::MultiFab& refreshedDiagonal() const
    {
        if (*diagonalDirty)
        {
            la::computeFaceCoeffDiag(exec, diagonal, alpha, upper, storedLower());
            *diagonalDirty = false;
        }
        return *diagonal;
    }

    /* @brief The preconditioner for `config`, from THIS matrix's own coefficients:
     *        none / gmg / gmg_kokkos / mlmg, never declined. `storedLower()` and not `lower` --
     *        the hierarchy wants the ALIASED low side, as op() does.
     */
    std::shared_ptr<const gko::LinOp> makePrecond(const SolverConfig& config) const
    {
        return makeFaceCoeffPrecond(
                   la::makeExecutor(exec),
                   globalRows(),
                   alpha,
                   upper,
                   storedLower(),
                   mesh,
                   bc,
                   config
        )
            .op;
    }

    const char* name() const { return "MFFaceCoeffs"; }

    bool isAssembled() const { return false; }

    // The only warning this format gets that its stored diagonal is stale -- the coefficient
    // fields are public, so nothing else can observe a write.
    void markStale() { *diagonalDirty = true; }

    void zero()
    {
        markStale();
        detail::zeroCoefficients(alpha, upper, lower);
    }

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
        : exec(executor), bc(bcs), mesh(std::move(meshLevel)),
          diagonalDirty(std::make_shared<bool>(true))
    {
        detail::allocateCoefficients(mesh, sym, alpha, upper, lower);
        // Not zeroed: `diagonalDirty` starts true, so nothing reads it before it is written.
        diagonal = CellFieldLevel {std::make_shared<amrex::MultiFab>(mesh.ba, mesh.dm, 1, 0)};
    }

    // The row/column DIMENSION every rank must agree on -- the one place numPts() is right.
    gko::size_type globalRows() const { return static_cast<gko::size_type>(mesh.ba.numPts()); }
};

/* @class CsrMatrix
 * @brief Assembled face-coefficient format: the same coefficient fields, with op() returning
 *        the explicit Ginkgo Csr assembleFaceCoeffCsr builds from them. Single-box meshes
 *        only, which is assembleFaceCoeffCsr's restriction (sparse/csr.hpp).
 */
class CsrMatrix
{
public:

    // The fields below sit behind shared_ptr (MultiFab is not copyable), so a copy of the
    // format SHARES them and a write through one is seen through the other.
    NeoN::Executor exec {NeoN::SerialExecutor {}};
    la::BcArray bc {};
    // The layout the fields were ALLOCATED from, and nothing repoints them, so it is the one
    // source for ba/dm rather than a second one competing with alpha's.
    MeshLevel mesh;
    // The cell-centred diagonal SOURCE (ddt/Sp/reaction), not the assembled diagonal: the
    // negSumDiag fold is assembleFaceCoeffCsr's job.
    CellFieldLevel alpha;
    FaceFieldLevel upper;
    // No low side to WRITE when symmetric: an operator writing both would double every
    // coefficient. Storage still has one -- storedLower().
    std::optional<FaceFieldLevel> lower; // nullopt when symmetric

    static CsrMatrix
    symmetric(const NeoN::Executor& exec, MeshLevel mesh, const la::BcArray& bc = {})
    {
        return CsrMatrix(exec, std::move(mesh), Symmetry::symmetric, bc);
    }

    static CsrMatrix
    asymmetric(const NeoN::Executor& exec, MeshLevel mesh, const la::BcArray& bc = {})
    {
        return CsrMatrix(exec, std::move(mesh), Symmetry::asymmetric, bc);
    }

    // Assembles on the first call and after any write; otherwise hands back the matrix it
    // already built, so two op() calls with no write between return the SAME pointer.
    std::shared_ptr<const gko::LinOp> op() const
    {
        if (state_->dirty)
        {
            state_->csr = la::assembleFaceCoeffCsr(
                la::makeExecutor(exec),
                mesh,
                alpha,
                upper,
                // The STORED low side, aliasing upper when symmetric: the assembly reads
                // both sides of every face.
                storedLower(),
                // The caller's `bc`: it makes csr.cpp's side() DROP the boundary column
                // instead of emitting an explicit 0.0 at the wraparound one, and fold the
                // homogeneous domain BC onto the diagonal as `diag += sign*aFace`.
                bc
            );
            state_->dirty = false;
        }
        return state_->csr;
    }

    /* @brief none / mlmg only; gmg and gmg_kokkos are DECLINED, as the assembled solver
     *        always has -- every GMG path consumes the coefficient FIELDS through the
     *        matrix-free stencil. Null rather than a throw, so the CALLER writes the message.
     */
    std::shared_ptr<const gko::LinOp> makePrecond(const SolverConfig& config) const
    {
        if (config.precondKind != PrecondKind::none && config.precondKind != PrecondKind::mlmg)
        {
            return nullptr;
        }
        return makeMlmgPrecond(la::makeExecutor(exec), globalRows(), *alpha, config);
    }

    const char* name() const { return "CsrMatrix"; }

    bool isAssembled() const { return true; }

    // The only warning this format gets that its assembly is stale -- the coefficient fields
    // are public, so nothing else can observe a write, and there is no "done writing" call to
    // hook: a caller that then writes nothing pays one redundant assembly rather than needing
    // finalize().
    void markStale() { state_->dirty = true; }

    void zero()
    {
        markStale();
        detail::zeroCoefficients(alpha, upper, lower);
    }

    /* @brief The STORED low side, which always exists -- `upper` itself when symmetric, the
     *        alias the assembly's convention wants. Deliberately NOT `lower`'s reading, where a
     *        symmetric matrix has none; the differing types keep the two from being confused.
     */
    FaceFieldLevel storedLower() const { return lower.value_or(upper); }

    // Derived from `lower`, so no stored enum can drift from the storage.
    Symmetry symmetry() const { return lower ? Symmetry::asymmetric : Symmetry::symmetric; }

    // Rows this rank owns. localCount(), NOT numPts(): numPts() counts EVERY rank's cells.
    std::size_t localRows() const { return la::localCount(*alpha); }

private:

    /* @brief Shared with every copy, like the fields: copies address the same coefficients, so
     *        a per-object flag would let a write through one leave another handing out a stale
     *        matrix. A SLOT rather than flat members, because reassembly RE-SEATS `csr` with a
     *        new LinOp and a flat member assigned through one copy would leave the others on
     *        the old one.
     */
    struct Assembly
    {
        std::shared_ptr<const gko::LinOp> csr;
        bool dirty = true;
    };

    CsrMatrix(
        const NeoN::Executor& executor, MeshLevel meshLevel, Symmetry sym, const la::BcArray& bcs
    )
        : exec(executor), bc(bcs), mesh(std::move(meshLevel)), state_(std::make_shared<Assembly>())
    {
        detail::allocateCoefficients(mesh, sym, alpha, upper, lower);
    }

    // The row/column DIMENSION every rank must agree on -- the one place numPts() is right.
    gko::size_type globalRows() const { return static_cast<gko::size_type>(mesh.ba.numPts()); }

    std::shared_ptr<Assembly> state_;
};

static_assert(IsMatrix<MFFaceCoeffs>);
static_assert(IsMatrix<CsrMatrix>);

} // namespace blockamr::la
