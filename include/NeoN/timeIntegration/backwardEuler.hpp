// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/database/fieldCollection.hpp"
#include "NeoN/core/database/oldTimeCollection.hpp"
#include "NeoN/fields/field.hpp"
#include "NeoN/timeIntegration/timeIntegration.hpp"
#include "NeoN/dsl/solver.hpp"
#include "NeoN/linearAlgebra/linearSystem.hpp"
#include "NeoN/linearAlgebra/sparsityPattern.hpp"


namespace NeoN::timeIntegration
{

template<typename SolutionVectorType>
class BackwardEuler :
    public TimeIntegratorBase<SolutionVectorType>::template Register<
        BackwardEuler<SolutionVectorType>>
{

public:

    using ValueType = typename SolutionVectorType::VectorValueType;
    using Base = TimeIntegratorBase<SolutionVectorType>::template Register<
        BackwardEuler<SolutionVectorType>>;

    BackwardEuler(const Dictionary& schemeDict, const Dictionary& solutionDict)
        : Base(schemeDict, solutionDict)
    {}

    static std::string name() { return "backwardEuler"; }

    static std::string doc() { return "first order time integration method"; }

    static std::string schema() { return "none"; }

    void solve(dsl::Expression<ValueType>&, SolutionVectorType&, scalar, scalar) override
    {
        NF_ERROR_EXIT("Not implemented, use NeoN::dsl::detail::iterative_solve_impl");
    };

    std::unique_ptr<TimeIntegratorBase<SolutionVectorType>> clone() const override
    {
        return std::make_unique<BackwardEuler>(*this);
    }

    bool explicitIntegration() const override { return false; }
};


} // namespace NeoN
