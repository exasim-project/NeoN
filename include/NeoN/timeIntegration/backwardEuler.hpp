// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/database/fieldCollection.hpp"
#include "NeoN/core/database/oldTimeCollection.hpp"
#include "NeoN/fields/field.hpp"
#include "NeoN/timeIntegration/timeIntegration.hpp"
#include "NeoN/dsl/solver.hpp"

namespace NeoN::timeIntegration
{

namespace fvcc = NeoN::finiteVolume::cellCentred;

template<typename SolutionVectorType>
class BackwardEuler :
    public TimeIntegratorBase<SolutionVectorType>::template Register<
        BackwardEuler<SolutionVectorType>>
{

public:

    using ValueType = typename SolutionVectorType::VectorValueType;
    using Base = TimeIntegratorBase<SolutionVectorType>::template Register<
        BackwardEuler<SolutionVectorType>>;

    BackwardEuler(const Dictionary& timeIntegrationDict, const Dictionary& solutionDict)
        : Base(timeIntegrationDict, solutionDict)
    {}

    static std::string name() { return "backwardEuler"; }

    static std::string doc() { return "first order implicit time integration method"; }

    static std::string schema() { return "none"; }

    void solve(dsl::Expression<ValueType>& exp, SolutionVectorType& solution, scalar t, scalar dt)
        override
    {}

    std::unique_ptr<TimeIntegratorBase<SolutionVectorType>> clone() const override
    {
        return std::make_unique<BackwardEuler>(*this);
    }

    bool explicitIntegration() const override { return false; }
};

} // namespace NeoN
