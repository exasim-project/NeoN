// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2023 NeoN authors

#pragma once

#include "NeoN/core/database/fieldCollection.hpp"
#include "NeoN/core/database/oldTimeCollection.hpp"
#include "NeoN/fields/field.hpp"
#include "NeoN/timeIntegration/timeIntegration.hpp"

namespace NeoN::timeIntegration
{

template<typename SolutionVectorType>
class ForwardEuler :
    public TimeIntegratorBase<SolutionVectorType>::template Register<
        ForwardEuler<SolutionVectorType>>
{

public:

    using ValueType = typename SolutionVectorType::VectorValueType;
    using Base =
        TimeIntegratorBase<SolutionVectorType>::template Register<ForwardEuler<SolutionVectorType>>;

    ForwardEuler(const Dictionary& schemeDict, const Dictionary& solutionDict)
        : Base(schemeDict, solutionDict)
    {}

    static std::string name() { return "forwardEuler"; }

    static std::string doc() { return "first order time integration method"; }

    static std::string schema() { return "none"; }

    void solve(
        dsl::Expression<ValueType>& eqn,
        SolutionVectorType& solutionVector,
        [[maybe_unused]] scalar t,
        scalar dt
    ) override
    {
        auto source = eqn.explicitOperation(solutionVector.size());
        SolutionVectorType& oldSolutionVector =
            NeoN::finiteVolume::cellCentred::oldTime(solutionVector);

        solutionVector.internalVector() = oldSolutionVector.internalVector() - source * dt;
        solutionVector.correctBoundaryConditions();

        // check if executor is GPU
        if (std::holds_alternative<NeoN::GPUExecutor>(eqn.exec()))
        {
            Kokkos::fence();
        }
    };

    std::unique_ptr<TimeIntegratorBase<SolutionVectorType>> clone() const override
    {
        return std::make_unique<ForwardEuler>(*this);
    }
};


} // namespace NeoN
