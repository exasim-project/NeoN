// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/database/oldTimeCollection.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/ddtOperator.hpp"
#include "NeoN/core/dictionary.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
DdtOperator<ValueType>::~DdtOperator()
{}

template<typename ValueType>
DdtOperator<ValueType>::DdtOperator(dsl::Operator::Type termType, VolumeField<ValueType>& field)
    : dsl::OperatorMixin<VolumeField<ValueType>>(field.exec(), dsl::Coeff(1.0), field, termType) {};

template<typename ValueType>
void DdtOperator<ValueType>::explicitOperation(Vector<ValueType>& source, scalar, scalar dt) const
{
    const scalar dtInver = 1.0 / dt;
    const auto vol = this->getVector().mesh().cellVolumes().view();
    auto [sourceView, field, oldVector] =
        views(source, this->field_.internalVector(), oldTime(this->field_).internalVector());

    parallelFor(
        source.exec(),
        source.range(),
        NEON_LAMBDA(const localIdx celli) {
            sourceView[celli] += dtInver * (field[celli] - oldVector[celli]) * vol[celli];
        },
        "ddtOpertator::explicitOperation"
    );
}

template<typename ValueType>
void DdtOperator<ValueType>::bdf1Kernel(
    la::LinearSystem<ValueType, localIdx>& ls, scalar, scalar dt
) const
{
    const auto vol = this->getVector().mesh().cellVolumes().view();
    const auto operatorScaling = this->getCoefficient();
    const auto [diagOffs, oldVector] =
        views(ls.matrix().sparsity()->diagOffset(), oldTime(this->field_).internalVector());
    auto [matrix, rhs] = ls.view();

    const scalar a0a1 = 1.0 / dt;

    parallelFor(
        ls.exec(),
        {0, oldVector.size()},
        NEON_LAMBDA(const localIdx celli) {
            const auto idx = matrix.rowOffs[celli] + diagOffs[celli];
            const auto commonCoef = operatorScaling[celli] * vol[celli];
            matrix.values[idx] += commonCoef * a0a1 * one<ValueType>();
            rhs[celli] += commonCoef * a0a1 * oldVector[celli];
        },
        "ddtOperator::implicitOperation<BDF1>"
    );
}

template<typename ValueType>
void DdtOperator<ValueType>::bdf2Kernel(
    la::LinearSystem<ValueType, localIdx>& ls, scalar, scalar dt
) const
{
    const auto vol = this->getVector().mesh().cellVolumes().view();
    const auto operatorScaling = this->getCoefficient();
    auto& old = oldTime(this->field_);
    auto& oldOld = oldTime(old);
    const auto [diagOffs, oldVector, oldOldVector] =
        views(getSparsityPattern().diagOffset(), old.internalVector(), oldOld.internalVector());
    auto [matrix, rhs] = ls.view();

    const scalar a0 = 1.5 / dt;
    const scalar a1 = 2.0 / dt;
    const scalar a2 = -0.5 / dt;

    parallelFor(
        ls.exec(),
        {0, oldVector.size()},
        NEON_LAMBDA(const localIdx celli) {
            const auto idx = matrix.rowOffs[celli] + diagOffs[celli];
            const auto commonCoef = operatorScaling[celli] * vol[celli];
            matrix.values[idx] += commonCoef * a0 * one<ValueType>();
            rhs[celli] +=
                commonCoef * a1 * oldVector[celli] + commonCoef * a2 * oldOldVector[celli];
        },
        "ddtOperator::implicitOperation<BDF2>"
    );
}

template<typename ValueType>
void DdtOperator<ValueType>::implicitOperation(
    la::LinearSystem<ValueType, localIdx>& ls, scalar t, scalar dt
) const
{
    const int level = oldTimeLevel(this->field_);

    if (scheme_ == DdtScheme::BDF1)
    {
        bdf1Kernel(ls, t, dt);
    }
    else if (level < 2)
    {
        bdf1Kernel(ls, t, dt); // startup step
    }
    else
    {
        bdf2Kernel(ls, t, dt);
    }
}

template<typename ValueType>
void DdtOperator<ValueType>::read(const Input& input)
{
    if (!std::holds_alternative<NeoN::Dictionary>(input))
    {
        return;
    }

    const NeoN::Dictionary& dict = std::get<NeoN::Dictionary>(input);

    if (!dict.contains("ddtSchemes"))
    {
        return; // keep default BDF1
    }

    const Dictionary& ddtSchemes = dict.subDict("ddtSchemes");

    std::string schemeName;

    // Per-field override: ddt(fieldName)
    const std::string fieldKey = std::string("ddt(") + this->field_.name + ")";
    if (ddtSchemes.contains(fieldKey))
    {
        schemeName = ddtSchemes.get<std::string>(fieldKey);
    }

    // TODO (later: steadyState, CrankNicolson, etc.)
    if (schemeName == "BDF1")
    {
        scheme_ = DdtScheme::BDF1;
        return;
    }
    // static timeIntegration::BDF2 bdf2Scheme;
    if (schemeName == "BDF2")
    {
        scheme_ = DdtScheme::BDF2;
        return;
    }

    NF_ERROR_EXIT(std::format(
        "Unknown ddt scheme '{}' for field '{}'. Supported schemes are: BDF1, BDF2.",
        schemeName,
        this->field_.name
    ));
}

// instantiate the template class
template class DdtOperator<scalar>;
template class DdtOperator<Vec3>;

};
