// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2023 NeoN authors

#pragma once

#include <functional>

#include "NeoN/fields/field.hpp"
#include "NeoN/dsl/operator.hpp"
#include "NeoN/dsl/expression.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"

namespace NeoN::timeIntegration
{

/* @class Factory class to create time integration method by a given name
 * using NeoNs runTimeFactory mechanism
 */
template<typename SolutionType>
class TimeIntegratorBase :
    public RuntimeSelectionFactory<
        TimeIntegratorBase<SolutionType>,
        Parameters<const Dictionary&, const Dictionary&, const dsl::Operator::Type>>
{

public:

    using ValueType = typename SolutionType::VectorValueType;
    using Expression = NeoN::dsl::Expression<ValueType>;

    static std::string name() { return "timeIntegrationFactory"; }

    TimeIntegratorBase(
        const Dictionary& schemeDict, const Dictionary& solutionDict, const dsl::Operator::Type
    )
        : schemeDict_(schemeDict), solutionDict_(solutionDict)
    {}

    virtual ~TimeIntegratorBase() {}

    virtual void solve(
        Expression& eqn, SolutionType& sol, scalar t, scalar dt
    ) = 0; // Pure virtual function for solving

    // Pure virtual function for cloning
    virtual std::unique_ptr<TimeIntegratorBase> clone() const = 0;

protected:

    const Dictionary& schemeDict_;
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

    TimeIntegration(
        const Dictionary& schemeDict, const Dictionary& solutionDict, const dsl::Operator::Type type
    )
        : timeIntegratorStrategy_(TimeIntegratorBase<SolutionVectorType>::create(
            schemeDict.get<std::string>("type"), schemeDict, solutionDict, type
        )) {};

    void solve(Expression& eqn, SolutionVectorType& sol, scalar t, scalar dt)
    {
        timeIntegratorStrategy_->solve(eqn, sol, t, dt);
    }

private:

    std::unique_ptr<TimeIntegratorBase<SolutionVectorType>> timeIntegratorStrategy_;
};


} // namespace NeoN::timeIntegration
