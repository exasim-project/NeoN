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
#include <utility>

#include "NeoN/blockAmr/core/bc.hpp"
#include "NeoN/blockAmr/linearAlgebra/coefficients.hpp"
#include "NeoN/blockAmr/linearAlgebra/krylov/executor.hpp"
#include "NeoN/blockAmr/linearAlgebra/matrixFree/faceCoeffOp.hpp"
#include "NeoN/blockAmr/linearAlgebra/sparse/csr.hpp"
#include "NeoN/blockAmr/linearAlgebra/transfer.hpp"
#include "NeoN/core/executor/executor.hpp"

// The two concrete matrix FORMATS, both satisfying IsMatrix.
//
// They hold the SAME storage -- one cell-centred alpha plus the six face
// coefficient fields -- and hand out the same CellView/FaceView. That is the
// resolution of the question S2 left open (plans/blockamr-la-implementation.md
// §4.4): the narrow MultiFab-backed views stand for both formats and neither the
// views nor Matrix grows a second erasure. The design's objection to a
// MultiFab-backed CSR was that it would cost "a full extra copy of the
// coefficients per assembly"; it does not, because assembleFaceCoeffCsr already
// takes seven amrex::MultiFabs and reads them host-side. MultiFabs are this
// path's INPUT, not a staging buffer bolted on for the interface.
//
// The formats therefore differ in exactly one thing -- what op() hands back --
// which is why they live in one header instead of two.
//
// BOTH FORMATS CARRY A BcArray, AND SO DOES ops::Laplacian (S6b). Read this
// before changing either, because it looks like a double fold and is not.
//
// op() hands `bc` to the underlying machinery exactly as it did before S6b:
// FaceCoeffOp reflects the domain ghosts, assembleFaceCoeffCsr folds the same
// reflection onto the diagonal and drops the off-diagonal. S6b then ALSO made
// ops::Laplacian fold, into the coefficients it assembles: the boundary face
// coefficient becomes zero and (sign-1)*aF lands on the diagonal source, with
// -aF*scale*g on the rhs for an inhomogeneous datum (operators/laplacian.hpp).
//
// The two coexist because THE OPERATOR'S FOLD ZEROES aF, and every fold here is
// multiplicative in aF: the stencil computes aF*(sign*pC) = 0, and csr.cpp's
// side() does `diag += sign*aFace` = 0 on a column it then drops. So the second
// fold applies nothing. This was verified by building it, not by reasoning alone
// (S6b handoff §6.1 and §10).
//
// THE DEPENDENCY THAT MAKES IT SAFE, spelled out because it is load-bearing and
// invisible: an operator that folds onto the diagonal while LEAVING aF on the
// face breaks every non-periodic boundary, because then the fold below is applied
// to a live coefficient. The tripwire is measured -- that mutation reddens 19
// tests (S6b handoff §10) -- but note WHERE it is NOT:
// test_la_boundary_conditions.py::test_the_two_formats_agree_through_the_laplacian
// stays green through it, because both formats fold the live coefficient the same
// way and agree with each other while both are wrong. The guard that actually
// holds this line is the BITWISE coefficient assertion in the same file
// (test_laplacian_folds_the_boundary_into_the_coefficients); do not replace it
// with a solve-level comparison.
//
// Keeping `bc` here is also what preserves S6a's variable row length: with an
// all-zero BcArray, csr.cpp's else-branch (:109) emits an explicit 0.0 at the
// modular-wraparound column, so a non-periodic row would carry seven entries
// including a periodic coupling that does not exist.
//
// The legacy blockamr::la::FaceCoeffSolver path is UNCHANGED and folds BCs
// at apply time only: it shares its coefficient fields with the GMG hierarchy,
// which applies its own ghost reflection per level, so folding into them would
// apply every BC twice per level (plans/blockamr-la-implementation.md, "S6b --
// RESCOPED").

namespace blockamr::la
{

namespace detail
{

/* @brief The storage both face-coefficient formats own: alpha and the six face
 *        fields, plus the geometry, executor and domain BCs op() needs.
 *
 * The fields are held by std::shared_ptr because amrex::FabArray's copy
 * constructor is DELETED (AMReX_FabArray.H) -- a MultiFab cannot be a by-value
 * member of a copyable type. So a copied format SHARES its coefficient fields
 * with the original, which is also the only sane reading of a copy: both objects
 * then hand out the same CellView/FaceView pointers, and a write through one is
 * visible through the other.
 *
 * When the matrix is symmetric, lower[d] ALIASES upper[d]: the matrix-free
 * operator's documented symmetric convention is "pass the same MultiFab for u*
 * and l*" (matrixFree/faceCoeffOp.hpp), so aliasing is what the operators below
 * already expect, and coefficients() reports an empty `lower` view.
 */
struct FaceCoeffFields
{
    NeoN::Executor exec {NeoN::SerialExecutor {}};
    amrex::Geometry geom;
    la::BcArray bc {};
    Symmetry sym = Symmetry::symmetric;

    std::shared_ptr<amrex::MultiFab> alpha;
    std::array<std::shared_ptr<amrex::MultiFab>, 3> upper {};
    std::array<std::shared_ptr<amrex::MultiFab>, 3> lower {};

    static FaceCoeffFields make(
        const NeoN::Executor& exec,
        const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm,
        amrex::Geometry geom,
        Symmetry sym,
        const la::BcArray& bc
    )
    {
        FaceCoeffFields f;
        f.exec = exec;
        f.geom = std::move(geom);
        f.bc = bc;
        f.sym = sym;
        // MultiFabs are not zero-initialised (the arena recycles memory), so a
        // freshly built matrix is explicitly zeroed -- callers write only the
        // coefficients their operator contributes.
        f.alpha = std::make_shared<amrex::MultiFab>(ba, dm, 1, 0);
        f.alpha->setVal(0.0);
        for (int d = 0; d < 3; ++d)
        {
            const amrex::BoxArray fba = amrex::convert(ba, amrex::IntVect::TheDimensionVector(d));
            f.upper[static_cast<std::size_t>(d)] = std::make_shared<amrex::MultiFab>(fba, dm, 1, 0);
            f.upper[static_cast<std::size_t>(d)]->setVal(0.0);
            if (sym == Symmetry::asymmetric)
            {
                f.lower[static_cast<std::size_t>(d)] =
                    std::make_shared<amrex::MultiFab>(fba, dm, 1, 0);
                f.lower[static_cast<std::size_t>(d)]->setVal(0.0);
            }
            else
            {
                f.lower[static_cast<std::size_t>(d)] = f.upper[static_cast<std::size_t>(d)];
            }
        }
        return f;
    }

    // THE negSumDiag SEAM, as S7 left it. `diag` here is STILL `alpha`, the
    // cell-centred diagonal SOURCE (ddt/Sp/reaction) -- it is NOT the matrix
    // diagonal, and an operator writing through it writes a SOURCE.
    //
    // What changed in S7 is only where the matrix diagonal alpha -
    // (aE+aW+aN+aS+aT+aB) is computed: MFFaceCoeffs now STORES it in a field of
    // its own (MFFaceCoeffs::diagonal()) and the matrix-free stencils read that
    // field instead of re-deriving it per cell per apply. sparse/csr.cpp still
    // derives it inline while assembling (it is assembled once anyway), and the
    // GMG hierarchy still derives its own per level.
    //
    // MatrixCoefficients::diag was deliberately NOT repointed at the stored
    // diagonal: it is the write handle every operator uses, the stored diagonal
    // is a DERIVED quantity, and making the two the same name would mean an
    // operator's += landed on a value the matrix recomputes underneath it.
    MatrixCoefficients coefficients()
    {
        MatrixCoefficients c;
        c.diag = CellView {alpha.get()};
        c.upper = FaceView {{upper[0].get(), upper[1].get(), upper[2].get()}};
        if (sym == Symmetry::asymmetric)
        {
            c.lower = FaceView {{lower[0].get(), lower[1].get(), lower[2].get()}};
        }
        return c;
    }

    void zero()
    {
        alpha->setVal(0.0);
        for (int d = 0; d < 3; ++d)
        {
            upper[static_cast<std::size_t>(d)]->setVal(0.0);
            // Skipped when symmetric: lower[d] IS upper[d], already zeroed.
            if (sym == Symmetry::asymmetric)
            {
                lower[static_cast<std::size_t>(d)]->setVal(0.0);
            }
        }
    }

    // Rows this rank owns. localCount(), NOT boxArray().numPts(): numPts() counts
    // the cells on EVERY rank, and the two differ on more than one rank
    // (linearAlgebra/transfer.hpp).
    std::size_t localRows() const { return la::localCount(*alpha); }

    // The operators' row/column DIMENSION, which every rank must agree on -- the
    // one place numPts() is the right answer, exactly as the existing solvers use
    // it (persistent.cpp: KrylovSolver(exec, numPts(), localCount(*alpha))).
    gko::size_type globalRows() const
    {
        return static_cast<gko::size_type>(alpha->boxArray().numPts());
    }
};

} // namespace detail

/* @class MFFaceCoeffs
 * @brief Matrix-free face-coefficient format: op() is a FaceCoeffOp over the
 *        coefficient fields, and no MATRIX is ever assembled.
 *
 * One derived field is kept, the diagonal (see diagonal()); everything else is
 * evaluated per apply.
 *
 * Build one with symmetric()/asymmetric(), fill it through coefficients(), hand
 * it to a Matrix.
 */
class MFFaceCoeffs
{
public:

    // `bc` defaults to all-periodic (BcArray 0 == periodic), which is what the
    // four-argument form in the S4 brief means; a non-periodic matrix passes the
    // sides explicitly, exactly as FaceCoeffSolver's `bc` already does.
    static MFFaceCoeffs symmetric(
        const NeoN::Executor& exec,
        const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm,
        amrex::Geometry geom,
        const la::BcArray& bc = {}
    )
    {
        return MFFaceCoeffs(
            detail::FaceCoeffFields::make(exec, ba, dm, std::move(geom), Symmetry::symmetric, bc)
        );
    }

    static MFFaceCoeffs asymmetric(
        const NeoN::Executor& exec,
        const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm,
        amrex::Geometry geom,
        const la::BcArray& bc = {}
    )
    {
        return MFFaceCoeffs(
            detail::FaceCoeffFields::make(exec, ba, dm, std::move(geom), Symmetry::asymmetric, bc)
        );
    }

    // Built fresh per call rather than cached: the matrix-free operator holds
    // POINTERS to the coefficient fields on the device path, but stages pinned
    // copies of them on the host path, so a cached operator would go stale after
    // a write through coefficients() on exactly one of the two paths. Rebuilding
    // is unconditionally correct and this is a per-solve call, not a per-iteration
    // one. The DIAGONAL is what does not get rebuilt with it -- diagonal() below
    // is what makes it survive.
    std::shared_ptr<const gko::LinOp> op() const
    {
        return gko::share(la::FaceCoeffOp::create(
            la::makeExecutor(f_.exec),
            f_.exec,
            f_.alpha->boxArray(),
            f_.alpha->DistributionMap(),
            f_.geom,
            f_.globalRows(),
            f_.alpha.get(),
            f_.upper[0].get(),
            f_.lower[0].get(),
            f_.upper[1].get(),
            f_.lower[1].get(),
            f_.upper[2].get(),
            f_.lower[2].get(),
            // Still the caller's `bc`: the reflection it drives is inert on
            // coefficients an operator already folded (aF == 0 there), and it is
            // what a hand-written non-periodic coefficient set still needs. The
            // header says why that is not a double fold.
            f_.bc,
            nullptr,
            &diagonal()
        ));
    }

    /* @brief The stored fine-level matrix diagonal, alpha - (aE+aW+aN+aS+aT+aB).
     *
     * This is the S7 change: the matrix-free stencils read this field instead of
     * re-deriving the diagonal per cell per apply. It is a field of the MATRIX,
     * not of MatrixCoefficients -- coefficients().diag is still alpha, the
     * diagonal SOURCE an operator writes, and this is the derived quantity the
     * mat-vec consumes.
     *
     * Freshness works exactly as CsrMatrix's assembly does, and for the same
     * reason: `dirty` is set when the write handles are handed out (there is no
     * "done writing" call to hook), and cleared here. Recomputed lazily rather
     * than eagerly so a caller that takes the handles and writes nothing pays one
     * redundant pass, not a wrong answer. Shared with every copy of the matrix,
     * like the fields themselves, so a write through one copy cannot leave
     * another handing out a stale diagonal.
     *
     * No BC awareness, deliberately: domain BCs enter the mat-vec through the
     * ghost reflection, hence through the OFF-diagonal term, so alpha - sum(faces)
     * is BC-independent. That stayed true through S6b -- the fold it added moves a
     * boundary coefficient between the face fields and alpha, and this derivation
     * reads whatever those hold.
     */
    const amrex::MultiFab& diagonal() const
    {
        if (state_->dirty)
        {
            la::computeFaceCoeffDiag(
                f_.exec,
                *state_->diag,
                *f_.alpha,
                *f_.upper[0],
                *f_.lower[0],
                *f_.upper[1],
                *f_.lower[1],
                *f_.upper[2],
                *f_.lower[2]
            );
            state_->dirty = false;
        }
        return *state_->diag;
    }

    bool isAssembled() const { return false; }

    // Handing out write handles is the only warning this format gets that its
    // stored diagonal is about to be invalidated -- see diagonal().
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

    Symmetry symmetry() const { return f_.sym; }

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
        // Not zeroed: dirty starts true, so nothing can read it before
        // computeFaceCoeffDiag has written every valid cell.
        state_->diag = std::make_shared<amrex::MultiFab>(
            f_.alpha->boxArray(), f_.alpha->DistributionMap(), 1, 0
        );
    }

    detail::FaceCoeffFields f_;
    std::shared_ptr<Diagonal> state_;
};

/* @class CsrMatrix
 * @brief Assembled face-coefficient format: the same coefficient fields, with
 *        op() returning the explicit Ginkgo Csr assembleFaceCoeffCsr builds from
 *        them.
 *
 * Single-box meshes only, which is assembleFaceCoeffCsr's restriction, not a new
 * one (sparse/csr.hpp).
 */
class CsrMatrix
{
public:

    static CsrMatrix symmetric(
        const NeoN::Executor& exec,
        const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm,
        amrex::Geometry geom,
        const la::BcArray& bc = {}
    )
    {
        return CsrMatrix(
            detail::FaceCoeffFields::make(exec, ba, dm, std::move(geom), Symmetry::symmetric, bc)
        );
    }

    static CsrMatrix asymmetric(
        const NeoN::Executor& exec,
        const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm,
        amrex::Geometry geom,
        const la::BcArray& bc = {}
    )
    {
        return CsrMatrix(
            detail::FaceCoeffFields::make(exec, ba, dm, std::move(geom), Symmetry::asymmetric, bc)
        );
    }

    // Assembles on the first call and after any write; otherwise hands back the
    // matrix it already built. Two op() calls with no write between therefore
    // return the SAME pointer and assemble once.
    std::shared_ptr<const gko::LinOp> op() const
    {
        if (state_->dirty)
        {
            state_->csr = la::assembleFaceCoeffCsr(
                la::makeExecutor(f_.exec),
                f_.geom,
                *f_.alpha,
                *f_.upper[0],
                *f_.lower[0],
                *f_.upper[1],
                *f_.lower[1],
                *f_.upper[2],
                *f_.lower[2],
                // Still the caller's `bc`, and this is where it EARNS its keep:
                // it is what makes csr.cpp's side() drop the boundary column
                // instead of emitting an explicit 0.0 at the wraparound one
                // (S6a's variable row length). The fold it adds on top,
                // `diag += sign*aFace`, is zero on a folded coefficient set.
                f_.bc
            );
            state_->dirty = false;
        }
        return state_->csr;
    }

    bool isAssembled() const { return true; }

    // Handing out write handles is the only warning this format gets that its
    // assembly is about to be invalidated -- there is no later "done writing"
    // call to hook. So the flag is set here, pessimistically: a caller that takes
    // the handles and writes nothing pays one redundant assembly, which is the
    // whole cost of not needing a finalize() step.
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

    Symmetry symmetry() const { return f_.sym; }

    std::size_t localRows() const { return f_.localRows(); }

    const NeoN::Executor& executor() const { return f_.exec; }

private:

    // Shared with every copy, like the fields themselves: copies address the same
    // coefficients, so they must also agree on whether the assembly over those
    // coefficients is current. A per-object flag would let a write through one
    // copy leave another handing out a stale matrix.
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
