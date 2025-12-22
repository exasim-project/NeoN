// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/database/oldTimeCollection.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/ddtOperator.hpp"
#include "NeoN/core/dictionary.hpp"
#include "NeoN/timeIntegration/ddt/BDF1.hpp"
#include "NeoN/timeIntegration/ddt/BDF2.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
DdtOperator<ValueType>::~DdtOperator()
{}

template<typename ValueType>
DdtOperator<ValueType>::DdtOperator(dsl::Operator::Type termType, VolumeField<ValueType>& field)
    : dsl::OperatorMixin<VolumeField<ValueType>>(field.exec(), dsl::Coeff(1.0), field, termType),
      sparsityPattern_(la::SparsityPattern::readOrCreate(field.mesh())) {};

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
        KOKKOS_LAMBDA(const localIdx celli) {
            sourceView[celli] += dtInver * (field[celli] - oldVector[celli]) * vol[celli];
        },
        "ddtOpertator::explicitOperation"
    );
}

template<typename ValueType>
void DdtOperator<ValueType>::implicitOperation(
    la::LinearSystem<ValueType, localIdx>& ls, scalar, scalar dt
) const
{
    const auto vol = this->getVector().mesh().cellVolumes().view();
    const auto operatorScaling = this->getCoefficient();
    const auto [diagOffs, oldVector] =
        views(getSparsityPattern().diagOffset(), oldTime(this->field_).internalVector());
    auto [matrix, rhs] = ls.view();

    const bool useMultistep = (scheme_->nSteps() > 1) && (!firstTimeStep_);
    const bool useStartup = (scheme_->nSteps() > 1) && (firstTimeStep_);

    // Select coefficients for the single-step kernel:
    // - if startup (BDF2 first step), use a0Startup/a1Startup
    // - else use the scheme's normal a0/a1
    const scalar a0 = useStartup ? scheme_->a0Startup(dt) : scheme_->a0(dt);
    const scalar a1 = useStartup ? scheme_->a1Startup(dt) : scheme_->a1(dt);

    if (!useMultistep)
    {
        parallelFor(
            ls.exec(),
            {0, oldVector.size()},
            KOKKOS_LAMBDA(const localIdx celli) {
                const auto idx = matrix.rowOffs[celli] + diagOffs[celli];
                const auto commonCoef = operatorScaling[celli] * vol[celli];
                matrix.values[idx] += commonCoef * a0 * one<ValueType>();
                rhs[celli] += commonCoef * a1 * oldVector[celli];
            },
            "ddtOperator::implicitOperation<nSteps=1>"
        );
    }
    else
    {
        const auto oldOldVector = oldTime(oldTime(this->field_)).internalVector().view();
        const scalar a2 = scheme_->a2(dt);
        parallelFor(
            ls.exec(),
            {0, oldVector.size()},
            KOKKOS_LAMBDA(const localIdx celli) {
                const auto idx = matrix.rowOffs[celli] + diagOffs[celli];
                const auto commonCoef = operatorScaling[celli] * vol[celli];
                matrix.values[idx] += commonCoef * a0 * one<ValueType>();
                rhs[celli] +=
                    commonCoef * a1 * oldVector[celli] + commonCoef * a2 * oldOldVector[celli];
            },
            "ddtOperator::implicitOperation<nSteps=2>"
        );
    }
    firstTimeStep_ = false;
}

template<typename ValueType>
timeIntegration::BDF1 DdtOperator<ValueType>::DEFAULT_BDF1_SCHEME {};

template<typename ValueType>
void DdtOperator<ValueType>::read(const Input& input)
{
    if (!std::holds_alternative<NeoN::Dictionary>(input))
    {
        return;
    }

    const NeoN::Dictionary& dict = std::get<NeoN::Dictionary>(input);

    if (dict.contains("timeIntegration"))
    {
        const Dictionary& timeIntegrationDict = dict.subDict("timeIntegration");
        if (timeIntegrationDict.contains("type"))
        {
            const std::string timeIntegrationType = timeIntegrationDict.get<std::string>("type");
            if (timeIntegrationType == "backwardEuler")
            {
                this->type_ = dsl::Operator::Type::Implicit;
            }
            else if (timeIntegrationType == "forwardEuler" || timeIntegrationType == "Runge-Kutta" || timeIntegrationType == "rungeKutta")
            {
                this->type_ = dsl::Operator::Type::Explicit;
            }
        }
    }

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
        scheme_ = &DEFAULT_BDF1_SCHEME;
        return;
    }
    static timeIntegration::BDF2 bdf2Scheme;
    if (schemeName == "BDF2")
    {
        scheme_ = &bdf2Scheme;
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
