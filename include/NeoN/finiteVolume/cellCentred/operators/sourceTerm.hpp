// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/core/input.hpp"
#include "NeoFOAM/dsl/operator.hpp"
#include "NeoFOAM/linearAlgebra/linearSystem.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/linearAlgebra/sparsityPattern.hpp"

namespace NeoFOAM::finiteVolume::cellCentred
{


template<typename ValueType>
class SourceTerm : public dsl::OperatorMixin<VolumeField<ValueType>>
{

public:

    SourceTerm(
        dsl::Operator::Type termType,
        VolumeField<scalar>& coefficients,
        VolumeField<ValueType>& field
    );

    void explicitOperation(Field<ValueType>& source) const;

    void implicitOperation(la::LinearSystem<ValueType, localIdx>& ls) const;

    void build(const Input&) {}

    std::string getName() const { return "sourceTerm"; }

private:

    const VolumeField<scalar>& coefficients_;
    const std::shared_ptr<SparsityPattern> sparsityPattern_;
};


} // namespace NeoFOAM
