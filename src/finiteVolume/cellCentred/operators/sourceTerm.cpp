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
    const VolumeField<ValueType>& field,
    bool suSp
)
    : dsl::OperatorMixin<VolumeField<ValueType>>(field.exec(), dsl::Coeff {1.0}, field, termType),
      spCoeff_(&coefficients), suSp_(suSp) {};

template<typename ValueType>
SourceTerm<ValueType>::SourceTerm(
    dsl::Operator::Type termType, VolumeField<ValueType>& coefficients
)
    : dsl::OperatorMixin<VolumeField<ValueType>>(
        coefficients.exec(), dsl::Coeff {1.0}, coefficients, termType
    ),
      spCoeff_(nullptr), suSp_(false) {};

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
    if (!spCoeff_)
    {
        NF_ERROR_EXIT("Not implemented");
    }
    const auto operatorScaling = this->getCoefficient();
    const auto vol = spCoeff_->mesh().cellVolumes().view();
    const auto [coeff] = views(spCoeff_->internalVector());
    auto values = ls.matrix().values().view();
    const auto ma = ls.faceToMatrixAddress()->view(ls.matrix().sparsity()->rowOffs().view());

    if (suSp_)
    {
        // SuSp: the positive part of the coefficient goes implicitly on the diagonal,
        // the negative part explicitly to the rhs using the current field — matching
        // OpenFOAM fvm::SuSp (diag += V*max(c,0); source -= V*min(c,0)*field).
        auto rhs = ls.rhs().view();
        const auto [fieldView] = views(this->field_.internalVector());
        NeoN::parallelFor(
            ls.exec(),
            {0, coeff.size()},
            NEON_LAMBDA(const localIdx celli) {
                const scalar c = operatorScaling[celli] * coeff[celli];
                const scalar cPos = c > scalar(0) ? c : scalar(0);
                const scalar cNeg = c < scalar(0) ? c : scalar(0);
                values[ma.diagIdx(celli)] += cPos * vol[celli] * one<ValueType>();
                rhs[celli] += -cNeg * vol[celli] * fieldView[celli];
            },
            "SuSp::implicitOperation"
        );
        return;
    }

    // Sp implicit: diagonal += scaling * spCoeff * volume
    NeoN::parallelFor(
        ls.exec(),
        {0, coeff.size()},
        NEON_LAMBDA(const localIdx celli) {
            values[ma.diagIdx(celli)] +=
                operatorScaling[celli] * coeff[celli] * vol[celli] * one<ValueType>();
        },
        "Sp::implicitOperation"
    );
}


// instantiate the template class
template class SourceTerm<scalar>;
template class SourceTerm<Vec3>;
};
