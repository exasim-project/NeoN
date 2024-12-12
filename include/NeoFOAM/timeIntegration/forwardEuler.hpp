// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/timeIntegration/timeIntegration.hpp"

namespace NeoFOAM::timeIntegration
{

template<typename SolutionFieldType>
class ForwardEuler :
    public TimeIntegratorBase<SolutionFieldType>::template Register<ForwardEuler<SolutionFieldType>>
{

public:

    using Expression = NeoFOAM::dsl::Expression;
    using Base =
        TimeIntegratorBase<SolutionFieldType>::template Register<ForwardEuler<SolutionFieldType>>;

    ForwardEuler(const Dictionary& dict) : Base(dict) {}

    static std::string name() { return "forwardEuler"; }

    static std::string doc() { return "first order time integration method"; }

    static std::string schema() { return "none"; }

    void solve(Expression& eqn, SolutionType& sol, scalar t, scalar dt) override
    {
        auto source = eqn.explicitOperation(sol.size());

        sol.internalField() -= source * dt;
        sol.correctBoundaryConditions();

        // check if executor is GPU
        if (std::holds_alternative<NeoFOAM::GPUExecutor>(eqn.exec()))
        {
            Kokkos::fence();
        }
    };

    std::unique_ptr<TimeIntegratorBase<SolutionFieldType>> clone() const override
    {
        return std::make_unique<ForwardEuler>(*this);
    }
};


} // namespace NeoFOAM
