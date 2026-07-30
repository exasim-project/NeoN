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

/* @brief The storage both face-coefficient formats own: ONE MatrixCoefficients, plus the
 *        executor and domain BCs op() needs. The fields sit behind shared_ptr (MultiFab is not
 *        copyable), so a copy SHARES them and a write through one is seen through the other.
 */
struct FaceCoeffFields
{
    NeoN::Executor exec {NeoN::SerialExecutor {}};
    la::BcArray bc {};
    // `mc.mesh` is the layout make() ALLOCATED the fields from and nothing repoints them, so
    // it is the source for ba/dm rather than a second one competing with alpha's.
    MatrixCoefficients mc;

    // `mesh` by value and MOVED into mc; a const& would add a Geometry copy per matrix.
    static FaceCoeffFields
    make(const NeoN::Executor& exec, MeshLevel mesh, Symmetry sym, const la::BcArray& bc)
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
        FaceCoeffFields f;
        f.exec = exec;
        f.bc = bc;
        f.mc.diag = CellFieldLevel {zeroed(mesh.ba, mesh.dm)};
        FaceFieldLevel lower {};
        for (int d = 0; d < 3; ++d)
        {
            const auto i = static_cast<std::size_t>(d);
            const amrex::BoxArray fba =
                amrex::convert(mesh.ba, amrex::IntVect::TheDimensionVector(d));
            f.mc.upper.dir[i] = zeroed(fba, mesh.dm);
            if (asym)
            {
                lower.dir[i] = zeroed(fba, mesh.dm);
            }
        }
        if (asym)
        {
            f.mc.lower = lower;
        }
        f.mc.mesh = std::move(mesh);
        return f;
    }

    const MeshLevel& mesh() const { return mc.mesh; }

    // The negSumDiag seam: `mc.diag` is STILL `alpha`, the cell-centred diagonal SOURCE
    // (ddt/Sp/reaction), NOT the matrix diagonal alpha - (aE+aW+aN+aS+aT+aB) -- that is the
    // derived field MFFaceCoeffs::diagonal() below, which an operator's += must not land on.
    MatrixCoefficients coefficients() const { return mc; }

    /* @brief The STORED low side, which always exists -- `upper` itself when symmetric, the
     *        alias FaceCoeffOp's convention wants. Deliberately NOT coefficients()'s reading,
     *        where `lower` is ABSENT; the differing types keep the two from being confused.
     */
    FaceFieldLevel storedLower() const { return mc.lower.value_or(mc.upper); }

    // Derived from `mc.lower`, so no stored enum can drift from the storage.
    Symmetry symmetry() const
    {
        return mc.symmetric() ? Symmetry::symmetric : Symmetry::asymmetric;
    }

    void zero()
    {
        (*mc.diag).setVal(0.0);
        for (int d = 0; d < 3; ++d)
        {
            mc.upper[d].setVal(0.0);
            // Nothing separate to zero when symmetric.
            if (mc.lower.has_value())
            {
                (*mc.lower)[d].setVal(0.0);
            }
        }
    }

    // Rows this rank owns. localCount(), NOT numPts(): numPts() counts EVERY rank's cells.
    std::size_t localRows() const { return la::localCount(*mc.diag); }

    // The row/column DIMENSION every rank must agree on -- the one place numPts() is right.
    gko::size_type globalRows() const { return static_cast<gko::size_type>(mc.mesh.ba.numPts()); }
};

} // namespace detail

/* @class MFFaceCoeffs
 * @brief Matrix-free face-coefficient format: op() is a FaceCoeffOp over the coefficient
 *        fields and no MATRIX is ever assembled. One derived field is kept, the diagonal
 *        (see diagonal()). Build with symmetric()/asymmetric(), fill through coefficients().
 */
class MFFaceCoeffs
{
public:

    // `bc` defaults to all-periodic (BcArray 0 == periodic).
    static MFFaceCoeffs
    symmetric(const NeoN::Executor& exec, MeshLevel mesh, const la::BcArray& bc = {})
    {
        return MFFaceCoeffs(
            detail::FaceCoeffFields::make(exec, std::move(mesh), Symmetry::symmetric, bc)
        );
    }

    static MFFaceCoeffs
    asymmetric(const NeoN::Executor& exec, MeshLevel mesh, const la::BcArray& bc = {})
    {
        return MFFaceCoeffs(
            detail::FaceCoeffFields::make(exec, std::move(mesh), Symmetry::asymmetric, bc)
        );
    }

    // Built fresh per call, not cached: the operator stages PINNED COPIES of the coefficient
    // fields on the host path, so a cached one would go stale after a write through
    // coefficients() on that path. This is a per-solve call, not a per-iteration one.
    std::shared_ptr<const gko::LinOp> op() const
    {
        // PROTOTYPE (C1): no refresh -- the stencil recomputes the centre term inline, so the
        // diagonal passed below is unused.
        return gko::share(la::FaceCoeffOp::create(
            la::makeExecutor(f_.exec),
            f_.exec,
            f_.mesh(),
            f_.globalRows(),
            f_.mc.diag,
            f_.mc.upper,
            // The STORED lower, not the interface's: symmetric ALIASES upper here,
            // exactly FaceCoeffOp's convention; coefficients() reports it ABSENT.
            f_.storedLower(),
            // The caller's `bc`: the ghost reflection it drives is what applies the
            // homogeneous domain BC.
            f_.bc,
            nullptr,
            CellFieldLevel {state_->diag}
        ));
    }

    /* @brief The stored matrix diagonal alpha - (aE+aW+aN+aS+aT+aB): a field of the MATRIX,
     *        not of MatrixCoefficients, whose `diag` is still alpha. Recomputed lazily off
     *        `dirty`, which the write handles set. PROTOTYPE (C1): the stencils bypass it.
     */
    const amrex::MultiFab& diagonal() const
    {
        if (state_->dirty)
        {
            la::computeFaceCoeffDiag(
                f_.exec, CellFieldLevel {state_->diag}, f_.mc.diag, f_.mc.upper, f_.storedLower()
            );
            state_->dirty = false;
        }
        return *state_->diag;
    }

    /* @brief The preconditioner for `config`, from THIS matrix's own coefficients:
     *        none / gmg / gmg_kokkos / mlmg, never declined. `storedLower()` and not
     *        coefficients().lower -- the hierarchy wants the ALIASED low side, as op() does.
     */
    std::shared_ptr<const gko::LinOp> makePrecond(const SolverConfig& config) const
    {
        return makeFaceCoeffPrecond(
                   la::makeExecutor(f_.exec),
                   f_.globalRows(),
                   f_.mc.diag,
                   f_.mc.upper,
                   f_.storedLower(),
                   f_.mesh(),
                   f_.bc,
                   config
        )
            .op;
    }

    const char* name() const { return "MFFaceCoeffs"; }

    bool isAssembled() const { return false; }

    // The only warning this format gets that its stored diagonal is stale -- see diagonal().
    MatrixCoefficients coefficients()
    {
        state_->dirty = true;
        return f_.coefficients();
    }

    void zero()
    {
        state_->dirty = true;
        f_.zero();
    }

    Symmetry symmetry() const { return f_.symmetry(); }

    std::size_t localRows() const { return f_.localRows(); }

    const NeoN::Executor& executor() const { return f_.exec; }

private:

    // Shared with every copy, like the fields themselves; see diagonal().
    struct Diagonal
    {
        std::shared_ptr<amrex::MultiFab> diag;
        bool dirty = true;
    };

    explicit MFFaceCoeffs(detail::FaceCoeffFields f)
        : f_(std::move(f)), state_(std::make_shared<Diagonal>())
    {
        // Not zeroed: `dirty` starts true, so nothing reads it before it is written.
        state_->diag = std::make_shared<amrex::MultiFab>(f_.mesh().ba, f_.mesh().dm, 1, 0);
    }

    detail::FaceCoeffFields f_;
    std::shared_ptr<Diagonal> state_;
};

/* @class CsrMatrix
 * @brief Assembled face-coefficient format: the same coefficient fields, with op() returning
 *        the explicit Ginkgo Csr assembleFaceCoeffCsr builds from them. Single-box meshes
 *        only, which is assembleFaceCoeffCsr's restriction (sparse/csr.hpp).
 */
class CsrMatrix
{
public:

    static CsrMatrix
    symmetric(const NeoN::Executor& exec, MeshLevel mesh, const la::BcArray& bc = {})
    {
        return CsrMatrix(
            detail::FaceCoeffFields::make(exec, std::move(mesh), Symmetry::symmetric, bc)
        );
    }

    static CsrMatrix
    asymmetric(const NeoN::Executor& exec, MeshLevel mesh, const la::BcArray& bc = {})
    {
        return CsrMatrix(
            detail::FaceCoeffFields::make(exec, std::move(mesh), Symmetry::asymmetric, bc)
        );
    }

    // Assembles on the first call and after any write; otherwise hands back the matrix it
    // already built, so two op() calls with no write between return the SAME pointer.
    std::shared_ptr<const gko::LinOp> op() const
    {
        if (state_->dirty)
        {
            state_->csr = la::assembleFaceCoeffCsr(
                la::makeExecutor(f_.exec),
                f_.mesh(),
                f_.mc.diag,
                f_.mc.upper,
                // The STORED low side, aliasing upper when symmetric: the assembly reads
                // both sides of every face.
                f_.storedLower(),
                // The caller's `bc`: it makes csr.cpp's side() DROP the boundary column
                // instead of emitting an explicit 0.0 at the wraparound one, and fold the
                // homogeneous domain BC onto the diagonal as `diag += sign*aFace`.
                f_.bc
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
        return makeMlmgPrecond(la::makeExecutor(f_.exec), f_.globalRows(), *f_.mc.diag, config);
    }

    const char* name() const { return "CsrMatrix"; }

    bool isAssembled() const { return true; }

    // Handing out write handles is the only warning this format gets that its assembly is
    // stale -- there is no "done writing" call to hook, so the flag is set pessimistically:
    // a caller that writes nothing pays one redundant assembly instead of needing finalize().
    MatrixCoefficients coefficients()
    {
        state_->dirty = true;
        return f_.coefficients();
    }

    void zero()
    {
        state_->dirty = true;
        f_.zero();
    }

    Symmetry symmetry() const { return f_.symmetry(); }

    std::size_t localRows() const { return f_.localRows(); }

    const NeoN::Executor& executor() const { return f_.exec; }

private:

    // Shared with every copy, like the fields: copies address the same coefficients, so a
    // per-object flag would let a write through one leave another handing out a stale matrix.
    struct Assembly
    {
        std::shared_ptr<const gko::LinOp> csr;
        bool dirty = true;
    };

    explicit CsrMatrix(detail::FaceCoeffFields f)
        : f_(std::move(f)), state_(std::make_shared<Assembly>())
    {}

    detail::FaceCoeffFields f_;
    std::shared_ptr<Assembly> state_;
};

static_assert(IsMatrix<MFFaceCoeffs>);
static_assert(IsMatrix<CsrMatrix>);

} // namespace blockamr::la
