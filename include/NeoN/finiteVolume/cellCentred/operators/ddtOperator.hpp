// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#pragma once

#include "NeoN/fields/field.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/input.hpp"
#include "NeoN/dsl/spatialOperator.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/core/database/oldTimeCollection.hpp"
#include "NeoN/finiteVolume/cellCentred/linearAlgebra/sparsityPattern.hpp"

namespace NeoN::finiteVolume::cellCentred
{


template<typename ValueType>
class DdtOperator : public dsl::OperatorMixin<VolumeField<ValueType>>
{

public:

    using FieldValueType = ValueType;

    DdtOperator(dsl::Operator::Type termType, VolumeField<ValueType>& field);

    void explicitOperation(Field<ValueType>& source, scalar, scalar dt) const;

    void implicitOperation(la::LinearSystem<ValueType, localIdx>& ls, scalar, scalar dt);

    void build(const Input&) {}

    std::string getName() const { return "DdtOperator"; }

private:

    const std::shared_ptr<SparsityPattern> sparsityPattern_;
};


} // namespace NeoN
