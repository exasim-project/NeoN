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
    if (scheme_ == DdtScheme::SteadyState)
    {
        return;
    }
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

    if (schemeName == "steadyState")
    {
        scheme_ = DdtScheme::SteadyState;
        return;
    }
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

    NF_ERROR_EXIT(fmt::format(
        fmt::runtime("Unknown ddt scheme '{}' for field '{}'. Supported schemes are: steadyState, "
                     "BDF1, BDF2."),
        schemeName,
        this->field_.name
    ));
}

// instantiate the template class
template class DdtOperator<scalar>;
template class DdtOperator<Vec3>;

};
