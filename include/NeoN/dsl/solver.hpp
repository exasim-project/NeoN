// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <utility>
#include <concepts>

#include "NeoN/fields/field.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/logging.hpp"
#include "NeoN/core/input.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/dsl/expression.hpp"
#include "NeoN/dsl/optimizer.hpp"
#include "NeoN/timeIntegration/timeIntegration.hpp"

#include "NeoN/linearAlgebra/linearSystem.hpp"
#include "NeoN/linearAlgebra/solver.hpp"


namespace NeoN::dsl
{

namespace detail
{
template<typename VectorType, typename IndexType>
la::SolverStats iterativeSolveImpl(
    Expression<typename VectorType::ElementType>& exp,
    la::LinearSystem<typename VectorType::ElementType>& ls,
    VectorType& solution,
    scalar t,
    scalar dt,
    const Dictionary& fvSchemes,
    const Dictionary& fvSolution,
    std::vector<const PostAssemblyBase<typename VectorType::ElementType, IndexType>*> ps = {}
)
{
    auto optExp = optimize(exp);
    optExp.read(fvSchemes);
    optExp.assemble(t, dt, ls, ps);

    // TODO move that to expression explicit operation or
    // into functor ?
    // subtract the explicit source term from the rhs
    auto expTmp = optExp.explicitOperation(solution.mesh().nCells());
    auto [vol, expSource, rhs] = views(solution.mesh().cellVolumes(), expTmp, ls.rhs());
    parallelFor(
        solution.exec(),
        {0, rhs.size()},
        NEON_LAMBDA(const localIdx i) { rhs[i] -= expSource[i] * vol[i]; }
    );

    auto solver = la::Solver(solution.exec(), fvSolution);
    fence(solution.exec());

    // Do some sanity checks before trying to solve
    NF_ASSERT(ls.exec() == solution.exec(), "Executors are not the same");
    return solver.solve(ls, solution.internalVector());
}

template<typename VectorType, typename IndexType>
la::SolverStats iterativeSolveImpl(
    Expression<typename VectorType::ElementType>& exp,
    VectorType& solution,
    scalar t,
    scalar dt,
    const Dictionary& fvSolution,
    std::vector<const PostAssemblyBase<typename VectorType::ElementType, IndexType>*> ps = {}
)
{
    auto ls = exp.assemble(solution.mesh(), t, dt, ps);

    auto solver = la::Solver(solution.exec(), fvSolution);
    fence(solution.exec());
    return solver.solve(ls, solution.internalVector());
}
}

/* @brief solve an expression
 *
 * @param exp - Expression which is to be solved/updated.
 * @param solution - Solution field, where the solution will be 'written to'.
 * @param t - the time at the start of the time step.
 * @param dt - time step for the temporal integration
 * @param fvSchemes - Dictionary containing spatial operator and time  integration properties
 * @param fvSolution - Dictionary containing linear solver properties
 * @param p - A chainable functor that performs manipulations on the assembled system
 */
template<typename VectorType, typename IndexType>
la::SolverStats solve(
    Expression<typename VectorType::ElementType, IndexType>& exp,
    VectorType& solution,
    scalar t,
    scalar dt,
    const Dictionary& fvSchemes,
    const Dictionary& fvSolution,
    std::vector<const PostAssemblyBase<typename VectorType::ElementType, IndexType>*> p = {}
)
{
    if (exp.temporalOperators().size() == 0 && exp.spatialOperators().size() == 0)
    {
        NF_ERROR_EXIT("No temporal or implicit terms to solve.");
    }
    exp.read(fvSchemes);
    auto integrator = timeIntegration::TimeIntegration<VectorType>(
        fvSchemes.subDict("timeIntegration"), fvSolution
    );

    if (exp.temporalOperators().size() > 0 && integrator.explicitIntegration())
    {
        // integrate equations in time
        integrator.solve(exp, solution, t, dt);
        return {{.numIter = -1, .initResNorm = 0, .finalResNorm = 0, .solveTime = 0}};
    }
    else
    {
        return detail::iterativeSolveImpl(exp, solution, t, dt, fvSolution, p);
    }
}

// ---------------------------------------------------------------------------
// Matrix under-relaxation (post-assembly fused kernel)
// ---------------------------------------------------------------------------
//
// Componentwise helpers. NeoN's working tree does NOT provide the cmpt* /
// componentMultiply family from the feat/underRelax reference, so the few
// componentwise ops the kernel needs are defined inline here over Vec3[i]
// using Kokkos::abs / Kokkos::max. zero<T>()/one<T>() already exist and are
// reused as-is. copySign/componentCopySign are ported VERBATIM from the
// reference (their definitions are sound; only their kernel call site was
// dropped, which is the sign-drop regression this kernel must NOT reproduce).

//! @brief Returns |mag| carrying the sign of s (scalar overload).
KOKKOS_INLINE_FUNCTION scalar copySign(const scalar mag, const scalar s)
{
    return (s >= 0) ? mag : -mag;
}

//! @brief Componentwise sign-copy (scalar overload).
KOKKOS_INLINE_FUNCTION scalar componentCopySign(const scalar mag, const scalar s)
{
    return copySign(mag, s);
}

//! @brief Componentwise sign-copy (Vec3 overload — all 3 components).
KOKKOS_INLINE_FUNCTION Vec3 componentCopySign(const Vec3& mag, const Vec3& s)
{
    return Vec3(copySign(mag[0], s[0]), copySign(mag[1], s[1]), copySign(mag[2], s[2]));
}

//! @brief Componentwise magnitude (scalar overload).
KOKKOS_INLINE_FUNCTION scalar componentMag(const scalar value) { return Kokkos::abs(value); }

//! @brief Componentwise magnitude (Vec3 overload — keeps all 3 components, does
//!        NOT collapse to the L2 norm that the global mag(Vec3) returns).
KOKKOS_INLINE_FUNCTION Vec3 componentMag(const Vec3& value)
{
    return Vec3(Kokkos::abs(value[0]), Kokkos::abs(value[1]), Kokkos::abs(value[2]));
}

//! @brief Componentwise max (scalar overload).
KOKKOS_INLINE_FUNCTION scalar componentMax(const scalar lhs, const scalar rhs)
{
    return Kokkos::max(lhs, rhs);
}

//! @brief Componentwise max (Vec3 overload — per component).
KOKKOS_INLINE_FUNCTION Vec3 componentMax(const Vec3& lhs, const Vec3& rhs)
{
    return Vec3(
        Kokkos::max(lhs[0], rhs[0]), Kokkos::max(lhs[1], rhs[1]), Kokkos::max(lhs[2], rhs[2])
    );
}

//! @brief Componentwise (Hadamard) multiply (scalar overload).
KOKKOS_INLINE_FUNCTION scalar componentMultiply(const scalar lhs, const scalar rhs)
{
    return lhs * rhs;
}

//! @brief Componentwise (Hadamard) multiply (Vec3 overload — per component, NOT the dot
//!        product). Used to form the per-component diagCmpt source correction
//!        diagCmpt[c] * psi_prev[c] in the matrix under-relaxation kernel.
KOKKOS_INLINE_FUNCTION Vec3 componentMultiply(const Vec3& lhs, const Vec3& rhs)
{
    return Vec3(lhs[0] * rhs[0], lhs[1] * rhs[1], lhs[2] * rhs[2]);
}

/* @brief Apply equation (matrix) under-relaxation to an assembled LinearSystem.
 *
 * OpenFOAM-parity path: NeoN bakes boundary contributions permanently into
 * the CSR diagonal at assembly time, so the augmented diagonal
 * `D_aug = matrix.values[diagIdx(cell)]` already contains the boundary diagonal.
 * OpenFOAM `fvMatrix::relax(alpha)` divides the ENTIRE augmented diagonal (internal +
 * boundary) by alpha: it adds the boundary internalCoeffs to the diagonal, clamps for
 * dominance, then `D /= alpha` (the boundary is removed from the stored diag afterwards
 * but re-added un-divided at solve, netting the same augmented value). This kernel
 * therefore relaxes the augmented diagonal DIRECTLY — no internal-only reconstruction.
 * Per cell:
 *
 *   D_aug     : matrix.values[diagIdx(cell)]            (augmented, boundary-baked)
 *   D_dom     : max(mag(D_aug), sumMagOffDiag[cell])    (dominance clamp on the augmented diag)
 *   D_relaxed : componentCopySign(D_dom / alpha, D_aug) (scale whole augmented diag by 1/alpha)
 *   write     : matrix.values[diagIdx(cell)] = D_relaxed
 *   source    : rhs[cell] += (D_relaxed - D_aug) * psi_prev[cell]
 *
 * History (boundary-diagonal relaxation fix, 2026-06-16): an earlier path reconstructed the
 * internal-only diagonal (D_internal = D_aug + boundaryDiag), divided ONLY the internal part by
 * alpha, and re-added the boundary UN-divided (D_relaxed = D_internal/alpha - boundaryDiag). That
 * left the relaxed augmented diagonal too large by boundaryDiag*(1-alpha)/alpha at every
 * boundary-adjacent cell, corrupting rAU = V/diag and HbyA and diverging the PIMPLE outer loop.
 * The synthetic unit tests masked it (boundaryDiag identically zero) and the rAU oracle encoded
 * the same wrong formula; it surfaced only as the full neoPimpleFoam-vs-pimpleFoam parity
 * divergence.
 *
 * The source correction makes the relaxed system share the fixed point of the
 * unrelaxed system (the correction cancels at the converged solution). `psi_prev`
 * is the field value at solve entry (`solution.internalVector()` — no
 * separate snapshot). The negative-diagonal sign convention (operators atomic_sub
 * positive flux into the diagonal; `scaledInvDiagNegLUx` reads `V/diag[0]` raw,
 * no mag()) is preserved componentwise via `componentCopySign` so `rAU`/`HbyA`
 * stay correct for alpha < 1.
 *
 * @param ls       The assembled LinearSystem (mutated in place on the augmented diagonal).
 * @param solution The solution VolumeField — supplies psi_prev (internalVector) and the
 *                 mesh (boundaryMesh().faceOwners() for the boundary-diagonal owner).
 * @param alpha    The under-relaxation factor. alpha <= 0 or alpha == 1 is a bitwise no-op.
 */
// Overload set is templated on the FULL LinearSystem parameter pack (mirroring
// removeBoundaryContributions, linearSystem.hpp) rather than LinearSystem<ElementType>.
// This matters because the real momentum system assembled by
// the DSL is the SEGREGATED vector-solve form — a SCALAR matrix (MatrixValueType == scalar) with
// a Vec3 RHS (RHSValueType == Vec3) — not LinearSystem<Vec3>. The diagonal/off-diagonal/boundary
// algebra therefore runs in MatrixValueType (scalar here), while the source correction and field
// read run in RHSValueType (Vec3 here); the cross term (dRelaxed - dAug) * field is
// MatrixValueType * RHSValueType (scalar * Vec3 = Vec3), which the existing operator* already
// supports. The synthetic NeoN unit tests use createEmptyLinearSystem<Vec3>(mesh) where
// MatrixValueType == RHSValueType == Vec3, so both forms are now exercised.
template<
    typename VectorType,
    typename MatrixValueType,
    typename RHSValueType,
    typename SystemMatrixType,
    typename BoundaryMatrixType>
void applyMatrixRelaxation(
    la::LinearSystem<MatrixValueType, RHSValueType, SystemMatrixType, BoundaryMatrixType>& ls,
    const VectorType& solution,
    scalar alpha
)
{
    // alpha<=0 guard: bitwise no-op. MUST be first so neither
    // matrix.values nor rhs is touched when no relaxation is requested.
    if (alpha <= 0.0 || alpha == 1.0)
    {
        return;
    }

    NeoN::Logging::info("URF applied");

    const scalar invAlpha = 1.0 / alpha;

    // View mechanics copied from removeBoundaryContributions (linearSystem.hpp) — the
    // in-tree, sign-correct precedent for view + diagonal addressing.
    auto lsView = ls.view();
    auto& matrix = lsView.matrix;
    auto& rhs = lsView.rhs;
    const auto ma = ls.faceToMatrixAddress()->view(ls.matrix().sparsity()->rowOffs().view());
    const auto [rowOffs, colIdxs] = views(ls.matrix().rowOffs(), ls.matrix().colIdxs());
    const auto field = solution.internalVector().view();

    const localIdx nCells = field.size();

    // FUSED single-pass relaxation (PERF): compute the off-diagonal magnitude sum inline and apply
    // the dominance clamp + diagonal boost + source correction in ONE kernel, eliminating the
    // per-call O(nCells) `sumOff` scratch allocation (and its zero-fill) and the second kernel
    // launch. Each cell needs only its OWN row's off-diagonal sum — no cross-cell dependency — and
    // relaxation only ever writes a cell's own diagonal (matrix.values[diagIdx]) and rhs[celli].
    // The row walk skips the diagonal (colIdxs[idx] != celli), and a cell's diagonal write never
    // aliases another cell's off-diagonal storage, so fusing is bitwise-identical to the prior two
    // passes.
    //
    // Matches OpenFOAM fvMatrix::relax(alpha): OF adds the boundary internalCoeffs to the diagonal,
    // clamps for dominance, then `D /= alpha` with the boundary contribution INCLUDED in the
    // division (removed from the stored diag afterwards but re-added un-divided at solve via
    // addBoundaryDiag) — netting a relaxed AUGMENTED diagonal of D_aug / alpha. NeoN bakes the
    // boundary permanently into matrix.values[diagIdx], so the augmented diagonal is already in
    // hand: we relax dAug DIRECTLY, with NO internal-only reconstruction. (The earlier
    // reconstruct-internal / re-add-un-divided-boundary path left the relaxed diagonal too large by
    // boundaryDiag*(1-alpha)/alpha at every boundary-adjacent cell, corrupting rAU/HbyA and
    // diverging the PIMPLE outer loop — the boundary-diagonal relaxation bug.)
    //
    // The dominance clamp runs on the augmented diagonal vs the off-diagonal magnitude
    // sum; componentCopySign preserves dAug's sign so scaledInvDiagNegLUx (rAU = V/diag,
    // no mag()) stays correct for alpha < 1. The source correction rhs += (dRelaxed -
    // dAug)*psi_prev makes the relaxed system share the unrelaxed fixed point and
    // exactly preserves the residual at psi_prev. Off-diagonal sum + diagonal algebra are in
    // MatrixValueType; the source-correction cross term (dRelaxed - dAug)*field is MatrixValueType
    // * RHSValueType (scalar
    // * Vec3 in the segregated momentum form).
    parallelFor(
        ls.exec(),
        {0, nCells},
        NEON_LAMBDA(const localIdx celli) {
            // Off-diagonal magnitude sum — cell-based CSR row gather, NO atomics (perf lever).
            auto sumOff = zero<MatrixValueType>();
            for (localIdx idx = rowOffs[celli]; idx < rowOffs[celli + 1]; ++idx)
            {
                if (colIdxs[idx] != celli)
                {
                    sumOff = sumOff + componentMag(matrix.values[idx]);
                }
            }
            const auto diagIdx = ma.diagIdx(celli);
            const auto dAug = matrix.values[diagIdx];                   // augmented, boundary-baked
            const auto dDom = componentMax(componentMag(dAug), sumOff); // dominance clamp
            const auto dRelaxed =
                componentCopySign(dDom * invAlpha, dAug); // scale whole augmented diag by 1/alpha
            matrix.values[diagIdx] = dRelaxed;
            rhs[celli] = rhs[celli] + (dRelaxed - dAug) * field[celli]; // source corr
        },
        "applyMatrixRelaxation"
    );

    // Implicit transform-BC per-component diagonal correction (slip/symmetry `diagCmpt_`).
    // The ginkgo backend subtracts diagCmpt[cell][c] from the SHARED scalar diagonal for
    // solve-component c (solveImplicitTransformComponent), so the effective per-component
    // diagonal is `dAug - diagCmpt[c]`. OpenFOAM's fvMatrix::relax divides the WHOLE augmented
    // diagonal — INCLUDING the boundary internalCoeffs this term represents — by alpha. The
    // shared scalar diagonal was just boosted by 1/alpha above; diagCmpt must be boosted by the
    // SAME 1/alpha, otherwise the slip/symmetry normal damping is applied at full strength
    // against a 1/alpha-larger diagonal and the constraint is effectively weakened by a factor
    // alpha at every transform-patch-adjacent cell whenever the equation is under-relaxed
    // (neoPimpleFoam; for PISO alpha==1 this whole function early-returns). The matching
    // per-component source correction rhs[c] += (1 - 1/alpha) * diagCmpt_old[c] * psi_prev[c]
    // preserves the fixed point (it cancels at convergence), exactly mirroring the
    // (dRelaxed - dAug)*psi_prev shared-diagonal correction above. diagCmpt is NOT folded into
    // the scalar dominance clamp (the shared dAug clamp already guarantees dominance and the
    // positive diagCmpt term only strengthens it; a per-component clamp is impossible against a
    // shared scalar diagonal). Runs for BOTH the segregated (scalar matrix / Vec3 rhs ->
    // diagCmpt is Vec3) and coupled multi-RHS (Vec3 matrix) instantiations; the guard skips it
    // (nullptr / scalar pressure solve) when no implicit transform patch exists, so the
    // neoIcoFoam/neoPisoFoam/pressure hot paths are untouched.
    if (ls.diagCmpt() && ls.diagCmpt()->size() > 0)
    {
        auto diagCmptV = ls.diagCmpt()->view();
        parallelFor(
            ls.exec(),
            {0, nCells},
            NEON_LAMBDA(const localIdx celli) {
                const auto dcOld = diagCmptV[celli];
                rhs[celli] = rhs[celli] + componentMultiply(dcOld, field[celli]) * (1.0 - invAlpha);
                diagCmptV[celli] = dcOld * invAlpha;
            },
            "applyMatrixRelaxationDiagCmpt"
        );
    }
}

// ---------------------------------------------------------------------------
// Field (explicit) under-relaxation
// ---------------------------------------------------------------------------

/* @brief Apply explicit field under-relaxation to a solution field's internal vector.
 *
 * Computes the OpenFOAM `relax()` blend on the internal field only:
 *
 *   psi[c] = prev[c] + alpha * (psi[c] - prev[c])
 *
 * `prev` is the caller-owned previous-iteration snapshot (e.g. from
 * `fieldRelaxationSnapshot`, taken at the top of the outer corrector). This is a
 * stateless free function mirroring `applyMatrixRelaxation`; NeoN stays
 * OpenFOAM-agnostic and sees only the scalar `alpha`.
 *
 * Boundary scope: only the internal vector is blended. The caller is
 * responsible for calling `solution.correctBoundaryConditions()` afterwards so the
 * boundary values are re-derived from the boundary conditions. (OpenFOAM's `relax()`
 * blends boundary values directly; both paths agree at the fixed point and for
 * `fixedValue` patches.)
 *
 * `alpha <= 0` or `alpha == 1` is a bitwise no-op (the early return touches nothing).
 * This is REQUIRED so the final outer iteration (`alpha == 1`) and an unset
 * factor leave the field byte-for-byte unchanged, side-stepping the `alpha == 1` ULP
 * round-trip trap of `prev + 1*(cur - prev)`.
 *
 * @param solution The solution VolumeField — its internal vector is blended in place.
 * @param previous The previous-iteration snapshot of the internal vector (deep copy).
 * @param alpha    The under-relaxation factor. alpha <= 0 or alpha == 1 is a bitwise no-op.
 */
template<typename VectorType>
void applyFieldRelaxation(
    VectorType& solution, const Vector<typename VectorType::ElementType>& previous, scalar alpha
)
{
    // Bitwise no-op guard: MUST be first so the internal vector is not touched when no
    // relaxation is requested (final iteration / unset factor, alpha==1 ULP trap).
    if (alpha <= 0.0 || alpha == 1.0)
    {
        return;
    }

    NF_ASSERT(solution.size() == previous.size(), "applyFieldRelaxation: field/prev size mismatch");

    auto [current, prev] = views(solution.internalVector(), previous);
    parallelFor(
        solution.exec(),
        {0, solution.size()},
        NEON_LAMBDA(const localIdx celli) {
            current[celli] = prev[celli] + alpha * (current[celli] - prev[celli]);
        },
        "applyFieldRelaxation"
    );
}

/* @brief Snapshot a field's internal vector for use as the `previous` arg to
 *        `applyFieldRelaxation` (caller-owned prevIter primitive).
 *
 * Returns an independent on-executor deep copy of `field.internalVector()` via the
 * `Vector` copy constructor — subsequent mutation of `field` does not affect the
 * returned snapshot. Do NOT hand-roll a `parallelFor` copy; the copy ctor already
 * performs an executor-correct deep copy.
 */
template<typename VectorType>
[[nodiscard]] auto fieldRelaxationSnapshot(const VectorType& field)
{
    return Vector<typename VectorType::ElementType>(field.internalVector());
}

} // namespace dsl
