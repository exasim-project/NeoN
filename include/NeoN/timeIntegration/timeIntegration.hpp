// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>

#include "NeoN/fields/field.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/dsl/expression.hpp"
#include "NeoN/linearAlgebra/solver.hpp"

namespace NeoN::timeIntegration
{

/* @class Factory class to create time integration method by a given name
 * using NeoNs runTimeFactory mechanism
 */
template<typename SolutionType>
class TimeIntegratorBase :
    public RuntimeSelectionFactory<
        TimeIntegratorBase<SolutionType>,
        Parameters<const Dictionary&, const Dictionary&>>
{

public:

    using ValueType = typename SolutionType::VectorValueType;
    using Expression = NeoN::dsl::Expression<ValueType>;

    static std::string name() { return "timeIntegrationFactory"; }

    TimeIntegratorBase(const Dictionary& timeIntegrationDict, const Dictionary& solutionDict)
        : timeIntegrationDict_(timeIntegrationDict), solutionDict_(solutionDict)
    {}

    virtual ~TimeIntegratorBase() {}

    virtual la::SolverStats solve(
        Expression& eqn, SolutionType& sol, scalar t, scalar dt
    ) = 0; // Pure virtual function for solving

    // Pure virtual function for cloning
    virtual std::unique_ptr<TimeIntegratorBase> clone() const = 0;

    virtual bool explicitIntegration() const { return true; }

protected:

    const Dictionary& timeIntegrationDict_;
    const Dictionary& solutionDict_;
};

/**
 * @class Factory class to create time integration method by a given name
 * using NeoNs runTimeFactory mechanism
 *
 * @tparam SolutionVectorType Type of the solution field eg, volumeVector or just a plain Vector
 */
template<typename SolutionVectorType>
class TimeIntegration
{

public:

    using ValueType = typename SolutionVectorType::VectorValueType;
    using Expression = NeoN::dsl::Expression<ValueType>;

    TimeIntegration(const TimeIntegration& timeIntegrator)
        : timeIntegratorStrategy_(timeIntegrator.timeIntegratorStrategy_->clone()) {};

    TimeIntegration(TimeIntegration&& timeIntegrator)
        : timeIntegratorStrategy_(std::move(timeIntegrator.timeIntegratorStrategy_)) {};

    TimeIntegration(const Dictionary& timeIntegrationDict, const Dictionary& solutionDict)
        : timeIntegratorStrategy_(TimeIntegratorBase<SolutionVectorType>::create(
            timeIntegrationDict.get<std::string>("type"), timeIntegrationDict, solutionDict
        )) {};

    la::SolverStats solve(Expression& eqn, SolutionVectorType& sol, scalar t, scalar dt)
    {
        timeIntegratorStrategy_->solve(eqn, sol, t, dt);
    }

    bool explicitIntegration() const { return timeIntegratorStrategy_->explicitIntegration(); }

private:

    std::unique_ptr<TimeIntegratorBase<SolutionVectorType>> timeIntegratorStrategy_;
};


} // namespace NeoN::timeIntegration
