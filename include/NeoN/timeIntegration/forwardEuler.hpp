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

template<typename SolutionFieldType>
class ForwardEuler :
    public TimeIntegratorBase<SolutionFieldType>::template Register<ForwardEuler<SolutionFieldType>>
{

public:

    using ValueType = typename SolutionFieldType::FieldValueType;
    using Base =
        TimeIntegratorBase<SolutionFieldType>::template Register<ForwardEuler<SolutionFieldType>>;

    ForwardEuler(const Dictionary& schemeDict, const Dictionary& solutionDict)
        : Base(schemeDict, solutionDict)
    {}

    static std::string name() { return "forwardEuler"; }

    static std::string doc() { return "first order time integration method"; }

    static std::string schema() { return "none"; }

    void solve(
        dsl::Expression<ValueType>& eqn,
        SolutionFieldType& solutionField,
        [[maybe_unused]] scalar t,
        scalar dt
    ) override
    {
        auto source = eqn.explicitOperation(solutionField.size());
        SolutionFieldType& oldSolutionField =
            NeoN::finiteVolume::cellCentred::oldTime(solutionField);

        solutionField.internalField() = oldSolutionField.internalField() - source * dt;
        solutionField.correctBoundaryConditions();

        // check if executor is GPU
        if (std::holds_alternative<NeoN::GPUExecutor>(eqn.exec()))
        {
            Kokkos::fence();
        }
        oldSolutionField.internalField() = solutionField.internalField();
    };

    std::unique_ptr<TimeIntegratorBase<SolutionFieldType>> clone() const override
    {
        return std::make_unique<ForwardEuler>(*this);
    }
};


} // namespace NeoN
