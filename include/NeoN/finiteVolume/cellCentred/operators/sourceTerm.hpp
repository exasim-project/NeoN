// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/core/input.hpp"
#include "NeoN/dsl/operator.hpp"
#include "NeoN/linearAlgebra/linearSystem.hpp"
#include "NeoN/linearAlgebra/sparsityPattern.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"

namespace NeoN::finiteVolume::cellCentred
{


template<typename ValueType>
class SourceTerm : public dsl::OperatorMixin<VolumeField<ValueType>>
{

public:

    using VectorValueType = ValueType;

    SourceTerm(
        dsl::Operator::Type termType,
        VolumeField<scalar>& coefficients,
        VolumeField<ValueType>& field
    );

    ~SourceTerm();

    void explicitOperation(Vector<ValueType>& source) const;

    void implicitOperation(la::LinearSystem<ValueType, localIdx>& ls) const;

    void read(const Input&) {}

    std::string getName() const { return "sourceTerm"; }

private:

    const VolumeField<scalar>& coefficients_;
};


} // namespace NeoN
