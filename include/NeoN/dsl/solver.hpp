// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <memory>
#include <type_traits>
#include <utility>
#include <concepts>

#include "NeoN/fields/field.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/input.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/dsl/expression.hpp"
#include "NeoN/timeIntegration/timeIntegration.hpp"

#include "NeoN/linearAlgebra/linearSystem.hpp"
#include "NeoN/linearAlgebra/solver.hpp"
#include "NeoN/linearAlgebra/sparsityPattern.hpp"


namespace NeoN::dsl
{

namespace detail
{
struct RelaxationCache
{
    bool hasRelaxationFactors = false;
    std::unordered_map<std::string, scalar> equations;
    std::unordered_map<std::string, scalar> fields;
};

inline const RelaxationCache& relaxationCache(const Dictionary& fvSolution)
{
    static std::mutex mtx;
    static std::unordered_map<const Dictionary*, RelaxationCache> caches;

    std::lock_guard<std::mutex> lock(mtx);

    auto it = caches.find(&fvSolution);
    if (it != caches.end())
    {
        return it->second;
    }

    RelaxationCache cache;

    if (!fvSolution.contains("relaxationFactors"))
    {
        NeoN::Logging::info("No relaxationFactors dictionary found");
        return caches.emplace(&fvSolution, std::move(cache)).first->second;
    }

    const Dictionary& rf = fvSolution.subDict("relaxationFactors");
    cache.hasRelaxationFactors = true;

    NeoN::Logging::info("Reading relaxationFactors");

    if (rf.isDict("equations"))
    {
        const auto& eqs = rf.subDict("equations");
        for (const auto& k : eqs.keys())
            if (eqs.isType<scalar>(k))
	    {
                cache.equations.emplace(k, eqs.get<scalar>(k));
	        NeoN::Logging::info("URF eqn {} = {}", k, eqs.get<scalar>(k));
	    }
    }

    if (rf.isDict("fields"))
    {
        const auto& flds = rf.subDict("fields");
        for (const auto& k : flds.keys())
            if (flds.isType<scalar>(k))
	    {
		cache.fields.emplace(k, flds.get<scalar>(k));
	        NeoN::Logging::info("URF field {} = {}", k, flds.get<scalar>(k));
	    }
    }

    // OpenFOAM-compatible fallback: flat entries apply to both
    for (const auto& k : rf.keys())
        if (rf.isType<scalar>(k))
        {
            scalar v = rf.get<scalar>(k);
            cache.equations.emplace(k, v);
            cache.fields.emplace(k, v);
	    NeoN::Logging::info("URF field {} = {}", k, v);
        }

    return caches.emplace(&fvSolution, std::move(cache)).first->second;
}

inline std::optional<scalar> findRelaxationFactor(
    const Dictionary& fvSolution,
    const std::string& fieldName
)
{
    const auto& c = relaxationCache(fvSolution);
    if (!c.hasRelaxationFactors) return std::nullopt;

    auto it = c.equations.find(fieldName);
    return (it != c.equations.end()) ? std::optional<scalar>(it->second) : std::nullopt;
}

inline std::optional<scalar> findFieldRelaxationFactor(
    const Dictionary& fvSolution,
    const std::string& fieldName
)
{
    const auto& c = relaxationCache(fvSolution);
    if (!c.hasRelaxationFactors) return std::nullopt;

    auto it = c.fields.find(fieldName);
    return (it != c.fields.end()) ? std::optional<scalar>(it->second) : std::nullopt;
}

template<typename ValueType>
KOKKOS_INLINE_FUNCTION ValueType componentMultiply(const ValueType& lhs, const ValueType& rhs)
{
    return lhs * rhs;
}

template<>
KOKKOS_INLINE_FUNCTION Vec3 componentMultiply(const Vec3& lhs, const Vec3& rhs)
{
    return Vec3(lhs[0] * rhs[0], lhs[1] * rhs[1], lhs[2] * rhs[2]);
}

template<typename ValueType>
KOKKOS_INLINE_FUNCTION ValueType cmptMagValue(const ValueType& value);

template<>
KOKKOS_INLINE_FUNCTION scalar cmptMagValue(const scalar& value)
{
    return Kokkos::abs(value);
}

template<>
KOKKOS_INLINE_FUNCTION Vec3 cmptMagValue(const Vec3& value)
{
    return Vec3(Kokkos::abs(value[0]), Kokkos::abs(value[1]), Kokkos::abs(value[2]));
}

template<typename ValueType>
KOKKOS_INLINE_FUNCTION scalar cmptMaxValue(const ValueType& value);

template<>
KOKKOS_INLINE_FUNCTION scalar cmptMaxValue(const scalar& value)
{
    return value;
}

template<>
KOKKOS_INLINE_FUNCTION scalar cmptMaxValue(const Vec3& value)
{
    return Kokkos::max(value[0], Kokkos::max(value[1], value[2]));
}

template<typename ValueType>
KOKKOS_INLINE_FUNCTION scalar cmptMinValue(const ValueType& value);

template<>
KOKKOS_INLINE_FUNCTION scalar cmptMinValue(const scalar& value)
{
    return value;
}

template<>
KOKKOS_INLINE_FUNCTION scalar cmptMinValue(const Vec3& value)
{
    return Kokkos::min(value[0], Kokkos::min(value[1], value[2]));
}

template<typename ValueType>
KOKKOS_INLINE_FUNCTION ValueType cmptMax(const ValueType& lhs, const ValueType& rhs);

template<>
KOKKOS_INLINE_FUNCTION scalar cmptMax(const scalar& lhs, const scalar& rhs)
{
    return Kokkos::max(lhs, rhs);
}

template<>
KOKKOS_INLINE_FUNCTION Vec3 cmptMax(const Vec3& lhs, const Vec3& rhs)
{
    return Vec3(
        Kokkos::max(lhs[0], rhs[0]),
        Kokkos::max(lhs[1], rhs[1]),
        Kokkos::max(lhs[2], rhs[2])
    );
}

KOKKOS_INLINE_FUNCTION scalar copySign(const scalar mag, const scalar s)
{
    return (s >= 0) ? mag : -mag;
}

KOKKOS_INLINE_FUNCTION scalar componentCopySign(const scalar mag, const scalar s)
{
    return copySign(mag, s);
}

KOKKOS_INLINE_FUNCTION Vec3 componentCopySign(const Vec3& mag, const Vec3& s)
{
    return Vec3(copySign(mag[0], s[0]), copySign(mag[1], s[1]), copySign(mag[2], s[2]));
}

template<typename VectorType>
void applyMatrixRelaxation(
    const la::SparsityPattern& sp,
    la::LinearSystem<typename VectorType::ElementType, localIdx>& ls,
    const VectorType& solution,
    scalar alpha
)
{
    NeoN::Logging::info("URF applied");
    const scalar invAlpha = 1.0 / alpha;
    auto [matrix, rhs] = ls.view();
    auto& mtx = ls.matrix();
    const auto [diagOffs, field, rowOffs, colIdxs] =
        views(sp.diagOffset(), solution.internalVector(), mtx.rowOffs(), mtx.colIdxs());
    auto& bcCoeffs =
        ls.auxiliaryCoefficients().template get<
            la::BoundaryCoefficients<typename VectorType::ElementType, localIdx>>(
            "boundaryCoefficients"
        );
    auto sumOff = Vector<typename VectorType::ElementType>(
        ls.exec(),
        field.size(),
        zero<typename VectorType::ElementType>()
    );
    auto boundaryDiagMag = Vector<scalar>(ls.exec(), field.size(), 0.0);
    auto boundaryDiagMin = Vector<scalar>(ls.exec(), field.size(), 0.0);
    auto [sumOffValues, bcMatrixValues, bcCellIdxs, boundaryDiagMagValues, boundaryDiagMinValues] =
        views(sumOff, bcCoeffs.matrixValues, bcCoeffs.rhsIdxs, boundaryDiagMag, boundaryDiagMin);

    parallelFor(
        ls.exec(),
        {0, field.size()},
        NEON_LAMBDA(const localIdx rowi) {
            const auto rowStart = rowOffs[rowi];
            const auto rowEnd = rowOffs[rowi + 1];
            auto sum = zero<typename VectorType::ElementType>();
            for (localIdx idx = rowStart; idx < rowEnd; ++idx)
            {
                if (colIdxs[idx] != rowi)
                {
                    sum += cmptMagValue(matrix.values[idx]);
                }
            }
            sumOffValues[rowi] = sum;
        },
        "applyMatrixRelaxationSumOffDiag"
    );

    parallelFor(
        ls.exec(),
        {0, bcMatrixValues.size()},
        NEON_LAMBDA(const localIdx facei) {
            const auto celli = bcCellIdxs[facei];
            const auto magValue = cmptMaxValue(cmptMagValue(bcMatrixValues[facei]));
            Kokkos::atomic_add(&boundaryDiagMagValues[celli], magValue);
            Kokkos::atomic_add(&boundaryDiagMinValues[celli], cmptMinValue(bcMatrixValues[facei]));
        },
        "applyMatrixRelaxationBoundaryDiag"
    );

    parallelFor(
        ls.exec(),
        {0, field.size()},
        NEON_LAMBDA(const localIdx celli) {
            const auto diagIdx = matrix.rowOffs[celli] + diagOffs[celli];
            const auto diag = matrix.values[diagIdx];
            const auto diagWithBoundary =
                diag + boundaryDiagMagValues[celli] * one<typename VectorType::ElementType>();
            const auto domMag =
                cmptMax(cmptMagValue(diagWithBoundary), sumOffValues[celli]);
	    const auto dominantDiag = componentCopySign(domMag, diagWithBoundary);
            const auto relaxedDiag =
                dominantDiag * invAlpha - boundaryDiagMinValues[celli] * one<typename VectorType::ElementType>();
            matrix.values[diagIdx] = relaxedDiag;
            rhs[celli] += componentMultiply(relaxedDiag - diag, field[celli]);
        },
        "applyMatrixRelaxation"
    );
}

template<typename VectorType>
void applyFieldRelaxation(
    VectorType& solution,
    const Vector<typename VectorType::ElementType>& previous,
    scalar alpha
)
{

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

template<typename VectorType>
la::SolverStats iterativeSolveImpl(
    Expression<typename VectorType::ElementType>& exp,
    const la::SparsityPattern& sp,
    la::LinearSystem<typename VectorType::ElementType, localIdx>& ls,
    VectorType& solution,
    scalar t,
    scalar dt,
    const Dictionary& fvSchemes,
    const Dictionary& fvSolution,
    std::vector<PostAssemblyBase<typename VectorType::ElementType>> ps,
    std::optional<scalar> eqnUrf,
    std::optional<scalar> fieldUrf
)
{
    exp.read(fvSchemes);
    exp.assemble(t, dt, sp, ls, ps);

    // TODO move that to expression explicit operation or
    // into functor ?
    // subtract the explicit source term from the rhs
    auto expTmp = exp.explicitOperation(solution.mesh().nCells());
    auto [vol, expSource, rhs] = views(solution.mesh().cellVolumes(), expTmp, ls.rhs());
    parallelFor(
        solution.exec(),
        {0, rhs.size()},
        NEON_LAMBDA(const localIdx i) { rhs[i] -= expSource[i] * vol[i]; }
    );

    if (eqnUrf && *eqnUrf > 0.0 && *eqnUrf != 1.0)
    {
        applyMatrixRelaxation(sp, ls, solution, *eqnUrf);
    }

    auto prev = Vector<typename VectorType::ElementType>(solution.internalVector());
    auto solver = la::Solver(solution.exec(), fvSolution);
    fence(solution.exec());
    auto stats = solver.solve(ls, solution.internalVector());
    if (fieldUrf && *fieldUrf > 0.0 && *fieldUrf != 1.0)
    {
        applyFieldRelaxation(solution, prev, *fieldUrf);
    }
    return stats;
}

template<typename VectorType>
la::SolverStats iterativeSolveImpl(
    Expression<typename VectorType::ElementType>& exp,
    VectorType& solution,
    scalar t,
    scalar dt,
    const Dictionary& fvSolution,
    std::vector<PostAssemblyBase<typename VectorType::ElementType>> ps,
    std::optional<scalar> eqnUrf,
    std::optional<scalar> fieldUrf
)
{
    auto [sparsity, ls] = exp.assemble(solution.mesh(), t, dt, ps);

    // TODO move that to expression explicit operation or
    // into functor ?
    // subtract the explicit source term from the rhs
    auto expTmp = exp.explicitOperation(solution.mesh().nCells());
    auto [vol, expSource, rhs] = views(solution.mesh().cellVolumes(), expTmp, ls.rhs());
    parallelFor(
        solution.exec(),
        {0, rhs.size()},
        NEON_LAMBDA(const localIdx i) { rhs[i] -= expSource[i] * vol[i]; }
    );

    if (eqnUrf && *eqnUrf > 0.0 && *eqnUrf != 1.0)
    {
        applyMatrixRelaxation(sparsity, ls, solution, *eqnUrf);
    }

    auto prev = Vector<typename VectorType::ElementType>(solution.internalVector());
    auto solver = la::Solver(solution.exec(), fvSolution);
    fence(solution.exec());
    auto stats = solver.solve(ls, solution.internalVector());
    if (fieldUrf && *fieldUrf > 0.0 && *fieldUrf != 1.0)
    {
        applyFieldRelaxation(solution, prev, *fieldUrf);
    }
    return stats;
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
template<typename VectorType>
la::SolverStats solve(
    Expression<typename VectorType::ElementType>& exp,
    VectorType& solution,
    scalar t,
    scalar dt,
    const Dictionary& fvSchemes,
    const Dictionary& fvSolution,
    std::vector<PostAssemblyBase<typename VectorType::ElementType>> p = {},
    std::optional<scalar> eqnUrf = std::nullopt,
    std::optional<scalar> fieldUrf = std::nullopt
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
        return {.numIter = -1, .initResNorm = 0, .finalResNorm = 0, .solveTime = 0};
    }
    else
    {
        return detail::iterativeSolveImpl(exp, solution, t, dt, fvSolution, p, eqnUrf, fieldUrf);
    }
}

} // namespace dsl
