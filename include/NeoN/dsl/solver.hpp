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
    static RelaxationCache cache;
    static std::once_flag cacheFlag;

    std::call_once(cacheFlag, [&]() {
        if (!fvSolution.contains("relaxationFactors"))
        {
            return;
        }

        cache.hasRelaxationFactors = true;
        const Dictionary& relaxationFactors = fvSolution.subDict("relaxationFactors");

        if (relaxationFactors.isDict("equations"))
        {
            const Dictionary& equations = relaxationFactors.subDict("equations");
            for (const auto& key : equations.keys())
            {
                if (equations.isType<scalar>(key))
                {
                    cache.equations.emplace(key, equations.get<scalar>(key));
                }
            }
        }

        if (relaxationFactors.isDict("fields"))
        {
            const Dictionary& fields = relaxationFactors.subDict("fields");
            for (const auto& key : fields.keys())
            {
                if (fields.isType<scalar>(key))
                {
                    cache.fields.emplace(key, fields.get<scalar>(key));
                }
            }
        }

        for (const auto& key : relaxationFactors.keys())
        {
            if (relaxationFactors.isType<scalar>(key))
            {
                cache.equations.emplace(key, relaxationFactors.get<scalar>(key));
                cache.fields.emplace(key, relaxationFactors.get<scalar>(key));
            }
        }
    });

    return cache;
}

inline std::optional<scalar> findRelaxationFactor(
    const Dictionary& fvSolution,
    const std::string& fieldName
)
{
    const auto& cache = relaxationCache(fvSolution);
    if (!cache.hasRelaxationFactors)
    {
        return std::nullopt;
    }

    auto it = cache.equations.find(fieldName);
    if (it != cache.equations.end())
    {
        return it->second;
    }

    return std::nullopt;
}

inline std::optional<scalar> findFieldRelaxationFactor(
    const Dictionary& fvSolution,
    const std::string& fieldName
)
{
    const auto& cache = relaxationCache(fvSolution);
    if (!cache.hasRelaxationFactors)
    {
        return std::nullopt;
    }

    auto it = cache.fields.find(fieldName);
    if (it != cache.fields.end())
    {
        return it->second;
    }

    return std::nullopt;
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
    const Dictionary& fvSolution
)
{
    const auto relaxFactor = findRelaxationFactor(fvSolution, solution.name);
    if (!relaxFactor.has_value())
    {
        return;
    }

    const scalar alpha = relaxFactor.value();
    if (alpha <= 0.0 || alpha == 1.0)
    {
        return;
    }

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
    const Dictionary& fvSolution
)
{
    const auto relaxFactor = findFieldRelaxationFactor(fvSolution, solution.name);
    if (!relaxFactor.has_value())
    {
        return;
    }

    const scalar alpha = relaxFactor.value();
    if (alpha <= 0.0 || alpha == 1.0)
    {
        return;
    }

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
    std::vector<PostAssemblyBase<typename VectorType::ElementType>> ps
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

    applyMatrixRelaxation(sp, ls, solution, fvSolution);

    auto prev = Vector<typename VectorType::ElementType>(solution.internalVector());
    auto solver = la::Solver(solution.exec(), fvSolution);
    fence(solution.exec());
    auto stats = solver.solve(ls, solution.internalVector());
    applyFieldRelaxation(solution, prev, fvSolution);
    return stats;
}

template<typename VectorType>
la::SolverStats iterativeSolveImpl(
    Expression<typename VectorType::ElementType>& exp,
    VectorType& solution,
    scalar t,
    scalar dt,
    const Dictionary& fvSolution,
    std::vector<PostAssemblyBase<typename VectorType::ElementType>> ps
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

    applyMatrixRelaxation(sparsity, ls, solution, fvSolution);

    auto prev = Vector<typename VectorType::ElementType>(solution.internalVector());
    auto solver = la::Solver(solution.exec(), fvSolution);
    fence(solution.exec());
    auto stats = solver.solve(ls, solution.internalVector());
    applyFieldRelaxation(solution, prev, fvSolution);
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
    std::vector<PostAssemblyBase<typename VectorType::ElementType>> p = {}
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
        return detail::iterativeSolveImpl(exp, solution, t, dt, fvSolution, p);
    }
}

} // namespace dsl
