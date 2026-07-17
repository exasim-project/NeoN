// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/timeIntegration/timeIntegration.hpp"

namespace NeoN::timeIntegration
{

/* @class SteadyState
 * @brief Steady-state "time integration": no temporal advancement.
 *
 * Selected via fvSchemes.timeIntegration.type = "steadyState". The ddt
 * operator contributes nothing to the system and stability comes from equation
 * under-relaxation (SIMPLE). The solve() hook is therefore a no-op, mirroring
 * the other implicit integrators.
 */
template<typename SolutionVectorType>
class SteadyState :
    public TimeIntegratorBase<SolutionVectorType>::template Register<
        SteadyState<SolutionVectorType>>
{

public:

    using ValueType = typename SolutionVectorType::VectorValueType;
    using Base =
        TimeIntegratorBase<SolutionVectorType>::template Register<SteadyState<SolutionVectorType>>;

    SteadyState(const Dictionary& timeIntegrationDict, const Dictionary& solutionDict)
        : Base(timeIntegrationDict, solutionDict)
    {}

    static std::string name() { return "steadyState"; }

    static std::string doc() { return "steady-state pseudo time integration (no time term)"; }

    static std::string schema() { return "none"; }

    void solve(dsl::Expression<ValueType>&, SolutionVectorType&, scalar, scalar) override {}

    std::unique_ptr<TimeIntegratorBase<SolutionVectorType>> clone() const override
    {
        return std::make_unique<SteadyState>(*this);
    }

    bool explicitIntegration() const override { return false; }
};

} // namespace NeoN
