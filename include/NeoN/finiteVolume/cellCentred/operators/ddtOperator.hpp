// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#pragma once

#include "NeoN/core/vector/vector.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/input.hpp"
#include "NeoN/dsl/operator.hpp"
#include "NeoN/linearAlgebra/linearSystem.hpp"
#include "NeoN/linearAlgebra/sparsityPattern.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
class DdtOperator : public dsl::OperatorMixin<VolumeField<ValueType>>
{

public:

    using VectorValueType = ValueType;

    DdtOperator(dsl::Operator::Type termType, VolumeField<ValueType>& field);

    void explicitOperation(Vector<ValueType>& source, scalar, scalar dt) const;

    void implicitOperation(la::LinearSystem<ValueType, localIdx>& ls, scalar, scalar dt) const;

    void read(const Input&) {}

    const la::SparsityPattern& getSparsityPattern() const { return sparsityPattern_; }

    std::string getName() const { return "DdtOperator"; }

private:

    // NOTE ddtOperator does not have a FactoryClass
    const la::SparsityPattern& sparsityPattern_;
};


} // namespace NeoN
