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

    // Sp: source += scaling * coefficients * field  (implicit or explicit).
    // With suSp = true this is instead OpenFOAM's SuSp (sign-aware split): the
    // positive part of the coefficient is treated implicitly (added to the
    // diagonal) and the negative part explicitly (added to the rhs using the
    // current field), so a coefficient of either sign keeps the matrix diagonally
    // dominant. Used by the kOmegaSST cross-diffusion / dilatation terms.
    SourceTerm(
        dsl::Operator::Type termType,
        const VolumeField<scalar>& coefficients,
        const VolumeField<ValueType>& field,
        bool suSp = false
    );

    // Su: source += scaling * coefficients  (explicit only)
    SourceTerm(dsl::Operator::Type termType, VolumeField<ValueType>& coefficients);

    ~SourceTerm();

    void explicitOperation(Vector<ValueType>& source) const;

    void implicitOperation(la::LinearSystem<ValueType>& ls) const;

    void read(const Input&) {}

    std::string getName() const { return "sourceTerm"; }


    Dictionary getConfig() const { return {}; }

private:

    // Non-null for Sp mode. Null for Su mode (field_ from mixin IS the coefficient).
    const VolumeField<scalar>* spCoeff_;

    // When true (and spCoeff_ non-null) the implicit assembly is OpenFOAM SuSp.
    bool suSp_;
};


} // namespace NeoN::finiteVolume::cellCentred
