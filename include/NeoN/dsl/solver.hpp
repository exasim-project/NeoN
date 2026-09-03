// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <utility>
#include <concepts>
#include <optional>

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
    // [fence-audit] removed redundant fence: the assemble kernels and the Ginkgo solve run on the
    // same Kokkos stream (ginkgo.cpp createGkoExecutor threads it through), so the solve is already
    // ordered after the assembly -- no device-wide sync needed here.

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
    // [fence-audit] removed redundant fence (same Kokkos stream as the Ginkgo solve; see above).
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
std::optional<la::SolverStats> solve(
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
        return std::nullopt; // no linear solve was performed, so no stats to return
    }
    else
    {
        return detail::iterativeSolveImpl(exp, solution, t, dt, fvSolution, p);
    }
}

// ---------------------------------------------------------------------------
// Matrix under-relaxation (post-assembly fused kernel)
// ---------------------------------------------------------------------------

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

/* @brief Apply equation (matrix) under-relaxation to an assembled LinearSystem.
 *
 * NeoN bakes boundary contributions permanently into the CSR diagonal at assembly time,
 * so the augmented diagonal `D_aug = matrix.values[diagIdx(cell)]` already contains the
 * boundary diagonal. This kernel relaxes the augmented diagonal DIRECTLY. Per cell:
 *
 *   D_aug     : matrix.values[diagIdx(cell)]            (augmented, boundary-baked)
 *   D_dom     : max(mag(D_aug), sumMagOffDiag[cell])    (dominance clamp on the augmented diag)
 *   D_relaxed : componentCopySign(D_dom / alpha, D_aug) (scale whole augmented diag by 1/alpha)
 *   write     : matrix.values[diagIdx(cell)] = D_relaxed
 *   source    : rhs[cell] += (D_relaxed - D_aug) * psi_prev[cell]
 *
 * The source correction makes the relaxed system share the fixed point of the
 * unrelaxed system (the correction cancels at the converged solution). `psi_prev`
 * is the field value at solve entry (`solution.internalVector()` — no separate snapshot).
 * The negative-diagonal sign convention is preserved componentwise via `componentCopySign`
 * so `rAU`/`HbyA` stay correct for alpha < 1.
 *
 * @param ls       The assembled LinearSystem (mutated in place on the augmented diagonal).
 * @param solution The solution VolumeField — supplies psi_prev (internalVector).
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

    const scalar invAlpha = 1.0 / alpha;

    auto lsView = ls.view();
    auto& matrix = lsView.matrix;
    auto& rhs = lsView.rhs;
    const auto ma = ls.faceToMatrixAddress()->view(ls.matrix().sparsity()->rowOffs().view());
    const auto [rowOffs, colIdxs] = views(ls.matrix().rowOffs(), ls.matrix().colIdxs());
    const auto field = solution.internalVector().view();

    const localIdx nCells = field.size();

    // FUSED single-pass relaxation: compute the off-diagonal magnitude sum inline and apply
    // the dominance clamp + diagonal boost + source correction in ONE kernel, eliminating the
    // per-call O(nCells) `sumOff` scratch allocation. Each cell needs only its OWN row's
    // off-diagonal sum — no cross-cell dependency — and relaxation only ever writes a cell's own
    // diagonal (matrix.values[diagIdx]) and rhs[celli]. The row walk skips the diagonal
    // (colIdxs[idx] != celli), and a cell's diagonal write never aliases another cell's
    // off-diagonal storage.
    //
    // componentCopySign preserves dAug's sign so rAU = V/diag stays correct for alpha < 1.
    // The source correction rhs += (dRelaxed - dAug)*psi_prev makes the relaxed system share
    // the unrelaxed fixed point. Off-diagonal sum + diagonal algebra are in MatrixValueType;
    // the source-correction cross term (dRelaxed - dAug)*field is MatrixValueType * RHSValueType
    // (scalar * Vec3 in the segregated momentum form).
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
}

// ---------------------------------------------------------------------------
// Field (explicit) under-relaxation
// ---------------------------------------------------------------------------

/* @brief Apply explicit field under-relaxation to a solution field's internal vector.
 *
 * Blends the internal field toward the previous outer-iteration value:
 *
 *   psi[c] = prev[c] + alpha * (psi[c] - prev[c])
 *
 * `prev` is the caller-owned previous-iteration snapshot (e.g. from
 * `fieldRelaxationSnapshot`, taken at the top of the outer corrector).
 *
 * Only the internal vector is blended; the caller is responsible for calling
 * `solution.correctBoundaryConditions()` afterwards so boundary values are
 * re-derived from the boundary conditions.
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
