// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#pragma once

#include "NeoN/core/vector.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/input.hpp"
#include "NeoN/dsl/operator.hpp"
#include "NeoN/linearAlgebra/linearSystem.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/linearAlgebra/sparsityPattern.hpp"

namespace NeoN::finiteVolume::cellCentred
{


template<typename ValueType>
class DdtOperator : public dsl::OperatorMixin<VolumeField<ValueType>>
{

public:

    using VectorValueType = ValueType;

    DdtOperator(dsl::Operator::Type termType, VolumeField<ValueType>& field);

    void explicitOperation(Vector<ValueType>& source, scalar, scalar dt) const;

    void implicitOperation(la::LinearSystem<ValueType>& ls, scalar, scalar dt);

    void read(const Input&) {}

    std::string getName() const { return "DdtOperator"; }

private:

    const std::shared_ptr<SparsityPattern> sparsityPattern_;
};


} // namespace NeoN
