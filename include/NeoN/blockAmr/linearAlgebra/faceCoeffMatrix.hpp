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

// The two concrete matrix FORMATS, both satisfying IsMatrix.
//
// They hold the SAME storage -- one cell-centred alpha plus the six face
// coefficient fields -- and hand out the same CellFieldLevel/FaceFieldLevel.
// That is the resolution of the question S2 left open (plans/blockamr-la-implementation.md
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
// BOTH FORMATS CARRY A BcArray, AND SO DOES ops::Laplacian. Read this before
// changing either, because it looks like a double fold and is not.
//
// THESE FORMATS ARE THE ONLY PLACE THE HOMOGENEOUS DOMAIN BC IS APPLIED, and that
// is the whole of the arrangement. op() hands `bc` to the underlying machinery:
// FaceCoeffOp reflects the domain ghosts per apply, assembleFaceCoeffCsr folds the
// same reflection onto the diagonal per assembly and drops the off-diagonal.
// ops::Laplacian carries `bc` for two other reasons only -- it must know which
// domain faces have no second cell to average gamma over, and it must know which
// sides read an inhomogeneous datum -- and it deliberately leaves the boundary FACE
// COEFFICIENT live (operators/laplacian.hpp). Every fold here is multiplicative in
// that coefficient, so a live one is exactly what they need.
//
// THE DEPENDENCY THAT MAKES IT SAFE, spelled out because it is load-bearing and
// invisible: an operator that ALSO folded -- zeroing the boundary coefficient and
// putting (sign-1)*aF on the diagonal source -- would leave the folds below inert
// and the FINE matrix identical, which is why the arrangement can be got wrong and
// still pass a solve. It is wrong on the COARSE levels: the GMG hierarchy built by
// makePrecond coarsens `alpha` with gmgRestrict, an eight-child volume average
// correct only for a dx-INDEPENDENT density, and (sign-1)*aF is 2*gamma/dx^2. On
// the face, gmgCoarsenFace's 1/4 is the right law. Measured: fully-Dirichlet CG+GMG
// took 12/13/14 iterations at 64/128/256^3 with the operator folding, 8/8/8 without
// -- 1.7x slower and mesh-DEPENDENT. laplacian.cpp carries the full measurement.
//
// The guard is the BITWISE coefficient assertion in test_la_boundary_conditions.py
// (test_laplacian_writes_the_boundary_face_coefficient); do not replace it with a
// solve-level comparison. test_the_two_formats_agree_through_the_laplacian sees
// nothing -- both formats fold whatever they are handed the same way, so they agree
// with each other under either convention.
//
// Keeping `bc` here is also what preserves S6a's variable row length: with an
// all-zero BcArray, csr.cpp's else-branch (:109) emits an explicit 0.0 at the
// modular-wraparound column, so a non-periodic row would carry seven entries
// including a periodic coupling that does not exist.
//
// The legacy blockamr::la::FaceCoeffSolver path folds BCs at apply time in exactly
// the same way, and always has: it shares its coefficient fields with the GMG
// hierarchy, which applies its own ghost reflection per level, so folding into them
// would apply every BC twice per level (plans/blockamr-la-implementation.md, "S6b --
// RESCOPED"). The la:: path now agrees with it on the stored coefficients too.

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
 * then hand out the same CellFieldLevel/FaceFieldLevel handles, and a write
 * through one is visible through the other.
 *
 * When the matrix is symmetric, lower[d] ALIASES upper[d]: the matrix-free
 * operator's documented symmetric convention is "pass the same MultiFab for u*
 * and l*" (matrixFree/faceCoeffOp.hpp), so aliasing is what the operators below
 * already expect, and coefficients() reports `lower` as NULLOPT. storedLower()
 * is the accessor for that aliased storage; the two readings are kept apart
 * deliberately (coefficients.hpp).
 */
struct FaceCoeffFields
{
    NeoN::Executor exec {NeoN::SerialExecutor {}};
    // NOT a MeshLevel, and that is the deliberate half of this slice's grouping.
    // `ba`/`dm` are DERIVED from alpha wherever they are needed (mesh() below,
    // globalRows(), MFFaceCoeffs' diagonal allocation), because alpha is the field
    // the coefficients actually live on and is therefore the only authority on
    // their layout. Storing the mesh here as well would give this struct a second
    // copy of ba/dm that nothing keeps in step with alpha's own -- a matrix could
    // then report one layout and write through another, silently. `geom` is stored
    // because it is NOT derivable from a MultiFab; there is exactly one source for
    // each fact. A MeshLevel is assembled on demand, in mesh(), from both.
    amrex::Geometry geom;
    la::BcArray bc {};
    Symmetry sym = Symmetry::symmetric;

    std::shared_ptr<amrex::MultiFab> alpha;
    std::array<std::shared_ptr<amrex::MultiFab>, 3> upper {};
    std::array<std::shared_ptr<amrex::MultiFab>, 3> lower {};

    // `mesh` is taken BY VALUE and its geom MOVED, which is what the four format
    // factories above did with their by-value `amrex::Geometry geom`; a const&
    // would add a Geometry copy per matrix construction. ba/dm are refcounted
    // handles, so copying them into the parameter is a refcount bump.
    static FaceCoeffFields
    make(const NeoN::Executor& exec, MeshLevel mesh, Symmetry sym, const la::BcArray& bc)
    {
        FaceCoeffFields f;
        f.exec = exec;
        f.geom = std::move(mesh.geom);
        f.bc = bc;
        f.sym = sym;
        // MultiFabs are not zero-initialised (the arena recycles memory), so a
        // freshly built matrix is explicitly zeroed -- callers write only the
        // coefficients their operator contributes.
        f.alpha = std::make_shared<amrex::MultiFab>(mesh.ba, mesh.dm, 1, 0);
        f.alpha->setVal(0.0);
        for (int d = 0; d < 3; ++d)
        {
            const amrex::BoxArray fba =
                amrex::convert(mesh.ba, amrex::IntVect::TheDimensionVector(d));
            f.upper[static_cast<std::size_t>(d)] =
                std::make_shared<amrex::MultiFab>(fba, mesh.dm, 1, 0);
            f.upper[static_cast<std::size_t>(d)]->setVal(0.0);
            if (sym == Symmetry::asymmetric)
            {
                f.lower[static_cast<std::size_t>(d)] =
                    std::make_shared<amrex::MultiFab>(fba, mesh.dm, 1, 0);
                f.lower[static_cast<std::size_t>(d)]->setVal(0.0);
            }
            else
            {
                f.lower[static_cast<std::size_t>(d)] = f.upper[static_cast<std::size_t>(d)];
            }
        }
        return f;
    }

    /* @brief The level this matrix lives on, assembled from its two sources: the
     *        layout off alpha, the geometry off the member.
     *
     * Built on demand rather than stored -- see the note on `geom` above. The
     * layout half is exactly what MFFaceCoeffs::op() spelled by hand before this
     * slice (alpha->boxArray(), alpha->DistributionMap()), so this is a grouping,
     * not a new fact.
     */
    MeshLevel mesh() const { return MeshLevel {alpha->boxArray(), alpha->DistributionMap(), geom}; }

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
        MatrixCoefficients c {mesh(), CellFieldLevel {alpha}, FaceFieldLevel {upper}, std::nullopt};
        if (sym == Symmetry::asymmetric)
        {
            c.lower = storedLower();
        }
        return c;
    }

    /* @brief The STORED low side, which always exists -- aliasing upper when the
     *        matrix is symmetric.
     *
     * Deliberately NOT what coefficients() reports. That is the INTERFACE reading:
     * absent when symmetric, because for a symmetric format lower[d] IS upper[d]
     * and an operator writing both would double every coefficient. This is the
     * STORAGE reading, and it is what the matrix-free operator's documented
     * convention wants ("pass the same MultiFab for u* and l*",
     * matrixFree/faceCoeffOp.hpp) -- an alias, never an absence.
     *
     * Both are true of one object at once, which is why they are different TYPES:
     * std::optional<FaceFieldLevel> at the interface, plain FaceFieldLevel here,
     * so handing one to something that meant the other does not compile.
     */
    FaceFieldLevel storedLower() const { return FaceFieldLevel {lower}; }

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

    // Built fresh per call rather than cached: the matrix-free operator holds
    // POINTERS to the coefficient fields on the device path, but stages pinned
    // copies of them on the host path, so a cached operator would go stale after
    // a write through coefficients() on exactly one of the two paths. Rebuilding
    // is unconditionally correct and this is a per-solve call, not a per-iteration
    // one. The DIAGONAL is what does not get rebuilt with it -- diagonal() below
    // is what makes it survive.
    std::shared_ptr<const gko::LinOp> op() const
    {
        // PROTOTYPE (C1): no refresh -- the stencil recomputes the centre term
        // inline, so there is nothing derived to keep fresh.
        return gko::share(la::FaceCoeffOp::create(
            la::makeExecutor(f_.exec),
            f_.exec,
            // Layout off alpha, geometry off the member -- see FaceCoeffFields::mesh().
            f_.mesh(),
            f_.globalRows(),
            CellFieldLevel {f_.alpha},
            FaceFieldLevel {f_.upper},
            // The STORED lower, not the interface's: symmetric ALIASES upper here,
            // which is exactly FaceCoeffOp's documented convention. coefficients()
            // reports the same object ABSENT.
            f_.storedLower(),
            // The caller's `bc`, and this is where it EARNS its keep: the
            // reflection it drives is what applies the homogeneous domain BC, on
            // an operator-assembled and a hand-written coefficient set alike. The
            // header says why that is not a double fold.
            f_.bc,
            nullptr,
            CellFieldLevel {state_->diag}
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
     * is BC-independent. It reads whatever the two fields hold, so it is correct
     * under either convention for where the boundary term is stored -- and the
     * convention is that it stays on the face (operators/laplacian.hpp).
     */
    const amrex::MultiFab& diagonal() const
    {
        if (state_->dirty)
        {
            la::computeFaceCoeffDiag(
                f_.exec,
                CellFieldLevel {state_->diag},
                CellFieldLevel {f_.alpha},
                FaceFieldLevel {f_.upper},
                f_.storedLower()
            );
            state_->dirty = false;
        }
        return *state_->diag;
    }

    /* @brief The preconditioner for `config`, built from THIS matrix's own
     *        coefficients. none / gmg / gmg_kokkos / mlmg; never declines.
     *
     * This format builds every preconditioner the matrix-free path has ever had,
     * because it holds exactly what FaceCoeffSolver holds: alpha, the six face
     * fields, the geometry and the BcArray. The call below is the SAME call
     * FaceCoeffSolver's constructor makes, with the same arguments in the same
     * order (precond.cpp) -- the hierarchy this returns is the hierarchy that
     * path has always built, which is why moving the construction here is
     * bitwise neutral.
     *
     * `storedLower()` and not coefficients().lower: the hierarchy wants the
     * ALIASED low side a symmetric matrix stores, exactly as op() does, not the
     * interface's absent one. Same distinction, same reason.
     *
     * The Kokkos V-cycle handle FaceCoeffPrecond also carries is dropped: it
     * exists for solver="mpir", which wraps it a second time, and a Matrix is
     * asked for a preconditioner, not for a solver.
     */
    std::shared_ptr<const gko::LinOp> makePrecond(const SolverConfig& config) const
    {
        const FaceFieldLevel upper {f_.upper};
        const FaceFieldLevel lower = f_.storedLower();
        return makeFaceCoeffPrecond(
                   la::makeExecutor(f_.exec),
                   f_.globalRows(),
                   f_.alpha.get(),
                   &upper[0],
                   &lower[0],
                   &upper[1],
                   &lower[1],
                   &upper[2],
                   &lower[2],
                   f_.geom,
                   f_.bc,
                   config
        )
            .op;
    }

    const char* name() const { return "MFFaceCoeffs"; }

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
                // (S6a's variable row length), AND what folds the homogeneous
                // domain BC onto the diagonal, `diag += sign*aFace`, from the live
                // boundary coefficient ops::Laplacian leaves there.
                f_.bc
            );
            state_->dirty = false;
        }
        return state_->csr;
    }

    /* @brief none / mlmg only; gmg and gmg_kokkos are DECLINED.
     *
     * The restriction is not new and not this format's invention: the assembled
     * solver has only ever accepted 'none' or 'mlmg' (solverConfig.hpp's note on
     * PrecondKind, enforced in FaceCoeffCsrSolver's constructor). The reason is
     * the same one that puts makePrecond on the matrix at all -- the GMG
     * hierarchy rediscretises the coefficient FIELDS on coarser levels, and an
     * assembled CSR is a matrix-free operator's opposite: it has the fields, but
     * every path that consumes them (GmgPrecondT, KokkosGmgApply) is written
     * against the matrix-free stencil. Declining is honest; building a
     * matrix-free hierarchy behind an assembled matrix's back would be a second
     * operator with no test tying the two together.
     *
     * Null rather than a throw, so the CALLER names itself and the format in the
     * message (coefficients.hpp). precond='mlmg' with no precond_mlmg to wrap
     * also comes back null and is refused by the caller the same way -- unlike
     * FaceCoeffCsrSolver, this format has no separate "requires precond_mlmg"
     * wording to preserve, because nothing ever threw it here.
     */
    std::shared_ptr<const gko::LinOp> makePrecond(const SolverConfig& config) const
    {
        if (config.precondKind != PrecondKind::none && config.precondKind != PrecondKind::mlmg)
        {
            return nullptr;
        }
        return makeMlmgPrecond(la::makeExecutor(f_.exec), f_.globalRows(), *f_.alpha, config);
    }

    const char* name() const { return "CsrMatrix"; }

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
