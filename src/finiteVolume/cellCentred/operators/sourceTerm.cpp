// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/sourceTerm.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
SourceTerm<ValueType>::~SourceTerm()
{}

template<typename ValueType>
SourceTerm<ValueType>::SourceTerm(
    dsl::Operator::Type termType,
    const VolumeField<scalar>& coefficients,
    const VolumeField<ValueType>& field
)
    : dsl::OperatorMixin<VolumeField<ValueType>>(field.exec(), dsl::Coeff {1.0}, field, termType),
      spCoeff_(&coefficients) {};

template<typename ValueType>
SourceTerm<ValueType>::SourceTerm(
    dsl::Operator::Type termType, VolumeField<ValueType>& coefficients
)
    : dsl::OperatorMixin<VolumeField<ValueType>>(
        coefficients.exec(), dsl::Coeff {1.0}, coefficients, termType
    ),
      spCoeff_(nullptr) {};

template<typename ValueType>
void SourceTerm<ValueType>::explicitOperation(Vector<ValueType>& source) const
{
    auto operatorScaling = this->getCoefficient();
    if (spCoeff_)
    {
        // Sp: source += scaling * spCoeff * field
        auto [sourceView, fieldView, coeff] =
            views(source, this->field_.internalVector(), spCoeff_->internalVector());
        NeoN::parallelFor(
            source.exec(),
            source.range(),
            NEON_LAMBDA(const localIdx celli) {
                sourceView[celli] += operatorScaling[celli] * coeff[celli] * fieldView[celli];
            },
            "Sp::explicitOperation"
        );
    }
    else
    {
        // Su: source += scaling * coeff  (field_ IS the coefficient for Su)
        auto [sourceView, coeff] = views(source, this->field_.internalVector());
        NeoN::parallelFor(
            source.exec(),
            source.range(),
            NEON_LAMBDA(const localIdx celli) {
                sourceView[celli] += operatorScaling[celli] * coeff[celli];
            },
            "Su::explicitOperation"
        );
    }
}

template<typename ValueType>
void SourceTerm<ValueType>::implicitOperation(la::LinearSystem<ValueType>& ls) const
{
    // Explicit template argument, not implicitOperation(ls) -- that would resolve back to this
    // exact non-template overload (exact match beats template deduction) and recurse forever.
    implicitOperation<la::CSRMatrix<ValueType, localIdx>>(ls);
}


// instantiate the template class
template class SourceTerm<scalar>;
template class SourceTerm<Vec3>;
};
