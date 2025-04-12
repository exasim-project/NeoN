// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include <functional>

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoFOAM/dsl/expression.hpp"

namespace NeoFOAM::timeIntegration
{

/* @class Factory class to create time integration method by a given name
 * using NeoFOAMs runTimeFactory mechanism
 */
template<typename SolutionType>
class TimeIntegratorBase :
    public RuntimeSelectionFactory<
        TimeIntegratorBase<SolutionType>,
        Parameters<const Dictionary&, const Dictionary&>>
{

public:

    using ValueType = typename SolutionType::FieldValueType;
    using Expression = NeoFOAM::dsl::Expression<ValueType>;

    static std::string name() { return "timeIntegrationFactory"; }

    TimeIntegratorBase(const Dictionary& schemeDict, const Dictionary& solutionDict)
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
 * using NeoFOAMs runTimeFactory mechanism
 *
 * @tparam SolutionFieldType Type of the solution field eg, volumeField or just a plain Field
 */
template<typename SolutionFieldType>
class TimeIntegration
{

public:


    using ValueType = typename SolutionFieldType::FieldValueType;
    using Expression = NeoFOAM::dsl::Expression<ValueType>;

    TimeIntegration(const TimeIntegration& timeIntegrator)
        : timeIntegratorStrategy_(timeIntegrator.timeIntegratorStrategy_->clone()) {};

    TimeIntegration(TimeIntegration&& timeIntegrator)
        : timeIntegratorStrategy_(std::move(timeIntegrator.timeIntegratorStrategy_)) {};

    TimeIntegration(const Dictionary& schemeDict, const Dictionary& solutionDict)
        : timeIntegratorStrategy_(TimeIntegratorBase<SolutionFieldType>::create(
            schemeDict.get<std::string>("type"), schemeDict, solutionDict
        )) {};

    void solve(Expression& eqn, SolutionFieldType& sol, scalar t, scalar dt)
    {
        timeIntegratorStrategy_->solve(eqn, sol, t, dt);
    }

private:

    std::unique_ptr<TimeIntegratorBase<SolutionFieldType>> timeIntegratorStrategy_;
};


} // namespace NeoFOAM::timeIntegration
