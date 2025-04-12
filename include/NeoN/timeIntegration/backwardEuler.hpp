// SPDX-License-Identifier: MIT
//
// SPDX-FileCopyrightText: 2023 NeoN authors

#pragma once

#include "NeoN/core/database/fieldCollection.hpp"
#include "NeoN/core/database/oldTimeCollection.hpp"
#include "NeoN/fields/field.hpp"
#include "NeoN/timeIntegration/timeIntegration.hpp"
#include "NeoN/dsl/solver.hpp"

#include "NeoN/linearAlgebra/linearSystem.hpp"

// TODO decouple from fvcc
#include "NeoN/finiteVolume/cellCentred/linearAlgebra/sparsityPattern.hpp"


namespace NeoN::timeIntegration
{

template<typename SolutionFieldType>
class BackwardEuler :
    public TimeIntegratorBase<SolutionFieldType>::template Register<
        BackwardEuler<SolutionFieldType>>
{

public:

    using ValueType = typename SolutionFieldType::FieldValueType;
    using Base =
        TimeIntegratorBase<SolutionFieldType>::template Register<BackwardEuler<SolutionFieldType>>;

    BackwardEuler(const Dictionary& schemeDict, const Dictionary& solutionDict)
        : Base(schemeDict, solutionDict)
    {}

    static std::string name() { return "backwardEuler"; }

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
        SolutionFieldType& oldSolutionField = finiteVolume::cellCentred::oldTime(solutionField);

        // solutionField.internalField() = oldSolutionField.internalField() - source * dt;
        // solutionField.correctBoundaryConditions();
        // solve sparse matrix system
        using ValueType = typename SolutionFieldType::ElementType;

        // TODO decouple from fvcc specific implementation
        auto sparsity = NeoN::finiteVolume::cellCentred::SparsityPattern(solutionField.mesh());
        auto ls = la::createEmptyLinearSystem<
            ValueType,
            localIdx,
            finiteVolume::cellCentred::SparsityPattern>(sparsity);

        eqn.implicitOperation(ls);

        auto values = ls.matrix().values();
        eqn.implicitOperation(ls, t, dt);

        auto solver = NeoN::la::Solver(solutionField.exec(), this->solutionDict_);
        solver.solve(ls, solutionField.internalField());

        // check if executor is GPU
        if (std::holds_alternative<NeoN::GPUExecutor>(eqn.exec()))
        {
            Kokkos::fence();
        }
        oldSolutionField.internalField() = solutionField.internalField();
    };

    std::unique_ptr<TimeIntegratorBase<SolutionFieldType>> clone() const override
    {
        return std::make_unique<BackwardEuler>(*this);
    }
};


} // namespace NeoN
