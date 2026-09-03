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
DdtOperator<ValueType>::DdtOperator(
    dsl::Operator::Type termType, VolumeField<scalar>& rho, VolumeField<ValueType>& field
)
    : dsl::OperatorMixin<VolumeField<ValueType>>(field.exec(), dsl::Coeff(1.0), field, termType),
      rho_(&rho) {};

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
void DdtOperator<ValueType>::bdf1Kernel(la::LinearSystem<ValueType>& ls, scalar, scalar dt) const
{
    const auto vol = this->getVector().mesh().cellVolumes().view();
    const auto operatorScaling = this->getCoefficient();
    const auto oldVector = oldTime(this->field_).internalVector().view();
    auto [rhs, values] = views(ls.rhs(), ls.matrix().values());
    const auto ma = ls.faceToMatrixAddress()->view(ls.matrix().sparsity()->rowOffs().view());

    const scalar a0a1 = 1.0 / dt;

    if (rho_ == nullptr)
    {
        parallelFor(
            ls.exec(),
            {0, oldVector.size()},
            NEON_LAMBDA(const localIdx celli) {
                const auto commonCoef = operatorScaling[celli] * vol[celli];
                values[ma.diagIdx(celli)] += commonCoef * a0a1 * one<ValueType>();
                rhs[celli] += commonCoef * a0a1 * oldVector[celli];
            },
            "ddtOperator::implicitOperation<BDF1>"
        );
    }
    else
    {
        const auto rhoNew = rho_->internalVector().view();
        const auto rhoOld = oldTime(*rho_).internalVector().view();
        parallelFor(
            ls.exec(),
            {0, oldVector.size()},
            NEON_LAMBDA(const localIdx celli) {
                const auto commonCoef = operatorScaling[celli] * vol[celli];
                values[ma.diagIdx(celli)] += rhoNew[celli] * commonCoef * a0a1 * one<ValueType>();
                rhs[celli] += rhoOld[celli] * commonCoef * a0a1 * oldVector[celli];
            },
            "ddtOperator::implicitOperation<BDF1,rho>"
        );
    }
}

template<typename ValueType>
void DdtOperator<ValueType>::bdf2Kernel(la::LinearSystem<ValueType>& ls, scalar, scalar dt) const
{
    const auto vol = this->getVector().mesh().cellVolumes().view();
    const auto operatorScaling = this->getCoefficient();
    auto& old = oldTime(this->field_);
    auto& oldOld = oldTime(old);
    const auto [oldVector, oldOldVector] = views(old.internalVector(), oldOld.internalVector());
    auto [rhs, values] = views(ls.rhs(), ls.matrix().values());

    const auto ma = ls.faceToMatrixAddress()->view(ls.matrix().sparsity()->rowOffs().view());

    const scalar a0 = 1.5 / dt;
    const scalar a1 = 2.0 / dt;
    const scalar a2 = -0.5 / dt;

    if (rho_ == nullptr)
    {
        parallelFor(
            ls.exec(),
            {0, oldVector.size()},
            NEON_LAMBDA(const localIdx celli) {
                const auto commonCoef = operatorScaling[celli] * vol[celli];
                values[ma.diagIdx(celli)] += commonCoef * a0 * one<ValueType>();
                rhs[celli] +=
                    commonCoef * a1 * oldVector[celli] + commonCoef * a2 * oldOldVector[celli];
            },
            "ddtOperator::implicitOperation<BDF2>"
        );
    }
    else
    {
        const auto rhoNew = rho_->internalVector().view();
        auto& rhoOldF = oldTime(*rho_);
        const auto [rhoOld, rhoOldOld] =
            views(rhoOldF.internalVector(), oldTime(rhoOldF).internalVector());
        parallelFor(
            ls.exec(),
            {0, oldVector.size()},
            NEON_LAMBDA(const localIdx celli) {
                const auto commonCoef = operatorScaling[celli] * vol[celli];
                values[ma.diagIdx(celli)] += rhoNew[celli] * commonCoef * a0 * one<ValueType>();
                rhs[celli] += rhoOld[celli] * commonCoef * a1 * oldVector[celli]
                            + rhoOldOld[celli] * commonCoef * a2 * oldOldVector[celli];
            },
            "ddtOperator::implicitOperation<BDF2,rho>"
        );
    }
}

template<typename ValueType>
void DdtOperator<ValueType>::implicitOperation(la::LinearSystem<ValueType>& ls, scalar t, scalar dt)
    const
{
    if (scheme_ == DdtScheme::SteadyState)
    {
        return;
    }
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
void DdtOperator<ValueType>::bdf1KernelScalarMtx(
    la::LinearSystem<scalar, ValueType>& ls, scalar, scalar dt
) const
{
    const auto vol = this->getVector().mesh().cellVolumes().view();
    const auto operatorScaling = this->getCoefficient();
    const auto oldVector = oldTime(this->field_).internalVector().view();
    auto [rhs, values] = views(ls.rhs(), ls.matrix().values());
    const auto ma = ls.faceToMatrixAddress()->view(ls.matrix().sparsity()->rowOffs().view());

    const scalar a0a1 = 1.0 / dt;

    if (rho_ == nullptr)
    {
        parallelFor(
            ls.exec(),
            {0, oldVector.size()},
            NEON_LAMBDA(const localIdx celli) {
                const auto commonCoef = operatorScaling[celli] * vol[celli];
                // scalar diagonal coefficient shared across all rhs components
                values[ma.diagIdx(celli)] += commonCoef * a0a1;
                rhs[celli] += commonCoef * a0a1 * oldVector[celli];
            },
            "ddtOperator::implicitOperationScalarMtx<BDF1>"
        );
    }
    else
    {
        const auto rhoNew = rho_->internalVector().view();
        const auto rhoOld = oldTime(*rho_).internalVector().view();
        parallelFor(
            ls.exec(),
            {0, oldVector.size()},
            NEON_LAMBDA(const localIdx celli) {
                // density-weighted: rho_n on the (scalar) diagonal, rho_o on the rhs
                const auto commonCoef = operatorScaling[celli] * vol[celli];
                values[ma.diagIdx(celli)] += rhoNew[celli] * commonCoef * a0a1;
                rhs[celli] += rhoOld[celli] * commonCoef * a0a1 * oldVector[celli];
            },
            "ddtOperator::implicitOperationScalarMtx<BDF1,rho>"
        );
    }
}

template<typename ValueType>
void DdtOperator<ValueType>::bdf2KernelScalarMtx(
    la::LinearSystem<scalar, ValueType>& ls, scalar, scalar dt
) const
{
    const auto vol = this->getVector().mesh().cellVolumes().view();
    const auto operatorScaling = this->getCoefficient();
    auto& old = oldTime(this->field_);
    auto& oldOld = oldTime(old);
    const auto [oldVector, oldOldVector] = views(old.internalVector(), oldOld.internalVector());
    auto [rhs, values] = views(ls.rhs(), ls.matrix().values());

    const auto ma = ls.faceToMatrixAddress()->view(ls.matrix().sparsity()->rowOffs().view());

    const scalar a0 = 1.5 / dt;
    const scalar a1 = 2.0 / dt;
    const scalar a2 = -0.5 / dt;

    if (rho_ == nullptr)
    {
        parallelFor(
            ls.exec(),
            {0, oldVector.size()},
            NEON_LAMBDA(const localIdx celli) {
                const auto commonCoef = operatorScaling[celli] * vol[celli];
                // scalar diagonal coefficient shared across all rhs components
                values[ma.diagIdx(celli)] += commonCoef * a0;
                rhs[celli] +=
                    commonCoef * a1 * oldVector[celli] + commonCoef * a2 * oldOldVector[celli];
            },
            "ddtOperator::implicitOperationScalarMtx<BDF2>"
        );
    }
    else
    {
        const auto rhoNew = rho_->internalVector().view();
        auto& rhoOldF = oldTime(*rho_);
        const auto [rhoOld, rhoOldOld] =
            views(rhoOldF.internalVector(), oldTime(rhoOldF).internalVector());
        parallelFor(
            ls.exec(),
            {0, oldVector.size()},
            NEON_LAMBDA(const localIdx celli) {
                const auto commonCoef = operatorScaling[celli] * vol[celli];
                values[ma.diagIdx(celli)] += rhoNew[celli] * commonCoef * a0;
                rhs[celli] += rhoOld[celli] * commonCoef * a1 * oldVector[celli]
                            + rhoOldOld[celli] * commonCoef * a2 * oldOldVector[celli];
            },
            "ddtOperator::implicitOperationScalarMtx<BDF2,rho>"
        );
    }
}

template<typename ValueType>
template<typename F>
    requires(!std::is_same_v<F, scalar>)
void DdtOperator<ValueType>::implicitOperation(
    la::LinearSystem<scalar, ValueType>& ls, scalar t, scalar dt
) const
{
    if (scheme_ == DdtScheme::SteadyState)
    {
        return;
    }
    const int level = oldTimeLevel(this->field_);

    if (scheme_ == DdtScheme::BDF1)
    {
        bdf1KernelScalarMtx(ls, t, dt);
    }
    else if (level < 2)
    {
        bdf1KernelScalarMtx(ls, t, dt); // startup step
    }
    else
    {
        bdf2KernelScalarMtx(ls, t, dt);
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

// The scalar-matrix implicitOperation is a constrained member template (disabled for scalar
// fields), so it is not covered by the class instantiation above; instantiate it explicitly
// for the Vec3 segregated vector-solve form.
template void
DdtOperator<Vec3>::implicitOperation<Vec3>(la::LinearSystem<scalar, Vec3>&, scalar, scalar) const;

};
