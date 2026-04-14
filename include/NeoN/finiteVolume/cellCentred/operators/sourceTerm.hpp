// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/core/input.hpp"
#include "NeoN/dsl/operator.hpp"
#include "NeoN/linearAlgebra/linearSystem.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"

namespace NeoN::finiteVolume::cellCentred
{


template<typename ValueType>
class SourceTerm : public dsl::OperatorMixin<VolumeField<ValueType>>
{

public:

    using VectorValueType = ValueType;

    // Sp: source += scaling * coefficients * field  (implicit or explicit)
    SourceTerm(
        dsl::Operator::Type termType,
        const VolumeField<scalar>& coefficients,
        const VolumeField<ValueType>& field
    );

    // Su: source += scaling * coefficients  (explicit only)
    SourceTerm(dsl::Operator::Type termType, VolumeField<ValueType>& coefficients);

    ~SourceTerm();

    void explicitOperation(Vector<ValueType>& source) const;

    void implicitOperation(la::LinearSystem<ValueType>& ls) const;

    void read(const Input&) {}

    std::string getName() const { return "sourceTerm"; }

private:

    // Non-null for Sp mode. Null for Su mode (field_ from mixin IS the coefficient).
    const VolumeField<scalar>* spCoeff_;
};


} // namespace NeoN::finiteVolume::cellCentred
