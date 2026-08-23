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
        // Sp implicit: diagonal += scaling * spCoeff * volume
        const auto operatorScaling = this->getCoefficient();
        const auto vol = spCoeff_->mesh().cellVolumes().view();
        const auto [coeff] = views(spCoeff_->internalVector());
        auto values = ls.matrix().values().view();
        const auto ma = ls.matrix().faceToMatrixView();

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
};


} // namespace NeoN::finiteVolume::cellCentred
