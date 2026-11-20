// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
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
    // With suSp = true this is instead the sign-aware SuSp split: the
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

    // Format-generic Sp assembly, defined here (not sourceTerm.cpp) so callers can
    // instantiate it for any SystemMatrixType, e.g. ELLMatrix. Overloads (not replaces) the
    // non-template implicitOperation() below -- for the CSR default, that exact non-template
    // match wins over this template per normal overload resolution, so DSL callers are
    // unaffected; ELL (or any other format) callers deduce SystemMatrixType here directly.
    template<typename SystemMatrixType>
    void implicitOperation(la::LinearSystem<ValueType, ValueType, SystemMatrixType>& ls) const
    {
        if (!spCoeff_)
        {
            NF_ERROR_EXIT("Not implemented");
        }
        const auto operatorScaling = this->getCoefficient();
        const auto vol = spCoeff_->mesh().cellVolumes().view();
        const auto [coeff] = views(spCoeff_->internalVector());
        auto values = ls.matrix().values().view();
        const auto ma = ls.matrix().faceToMatrixView();

        if (suSp_)
        {
            // SuSp: the positive part of the coefficient goes implicitly on the diagonal, the
            // negative part explicitly to the rhs using the current field:
            //   diag += V*max(c,0);  source -= V*min(c,0)*field
            //
            // The split is taken on the RAW coefficient and the expression scaling is applied
            // to the result, so the assembled contribution stays linear in that scaling — the
            // same property every other operator has. Splitting the already-scaled product
            // instead would make a negative scaling swap the implicit and explicit branches
            // rather than negate the split term, so `susp(c, phi) - susp(c, phi)` would not
            // cancel (expression subtraction sets the scaling to -1, see dsl/expression.hpp).
            auto rhs = ls.rhs().view();
            const auto [fieldView] = views(this->field_.internalVector());
            NeoN::parallelFor(
                ls.exec(),
                {0, coeff.size()},
                NEON_LAMBDA(const localIdx celli) {
                    const scalar c = coeff[celli];
                    const scalar cPos = c > scalar(0) ? c : scalar(0);
                    const scalar cNeg = c < scalar(0) ? c : scalar(0);
                    const scalar scaling = operatorScaling[celli];
                    values[ma.diagIdx(celli)] += scaling * cPos * vol[celli] * one<ValueType>();
                    rhs[celli] += -scaling * cNeg * vol[celli] * fieldView[celli];
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

    void implicitOperation(la::LinearSystem<ValueType>& ls) const;

    void read(const Input&) {}

    std::string getName() const { return "sourceTerm"; }


    Dictionary getConfig() const { return {}; }

private:

    // Non-null for Sp mode. Null for Su mode (field_ from mixin IS the coefficient).
    const VolumeField<scalar>* spCoeff_;

    // When true (and spCoeff_ non-null) the implicit assembly uses the SuSp sign split.
    bool suSp_;
};


} // namespace NeoN::finiteVolume::cellCentred
