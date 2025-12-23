// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

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

    static std::string doc() { return "first order explicit time integration method"; }

    static std::string schema() { return "none"; }

    void solve(
        dsl::Expression<ValueType>& eqn, SolutionVectorType& solutionVector, scalar t, scalar dt
    ) override
    {
        auto source = Vector<ValueType>(eqn.exec(), solutionVector.size(), zero<ValueType>());

        // spatial explicit operators
        eqn.explicitOperation(source);

        // temporal explicit operators (ddt explicit evaluation)
        eqn.explicitOperation(source, t, dt);

        SolutionVectorType& oldSolutionVector =
            NeoN::finiteVolume::cellCentred::oldTime(solutionVector);

        solutionVector.internalVector() = oldSolutionVector.internalVector() - source * dt;
        solutionVector.correctBoundaryConditions();

        fence(eqn.exec());
    };

    std::unique_ptr<TimeIntegratorBase<SolutionVectorType>> clone() const override
    {
        return std::make_unique<ForwardEuler>(*this);
    }
};


} // namespace NeoN
