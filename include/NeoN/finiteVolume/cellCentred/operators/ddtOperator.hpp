// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/input.hpp"
#include "NeoN/dsl/operator.hpp"
#include "NeoN/linearAlgebra/linearSystem.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/ddtScheme.hpp"
#include "NeoN/core/database/oldTimeCollection.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
class DdtOperator : public dsl::OperatorMixin<VolumeField<ValueType>>
{

public:

    using VectorValueType = ValueType;

    DdtOperator(dsl::Operator::Type termType, VolumeField<ValueType>& field);

    /* @brief Density-weighted temporal operator ddt(rho, field): the diagonal uses the
     *        current density rho and the rhs uses oldTime(rho), giving the conservative
     *        (rho_n*field - rho_o*field_o)/dt form (as opposed to the single-coefficient
     *        rho_n/dt*(field - field_o)). rho must carry an old-time collection.
     */
    DdtOperator(
        dsl::Operator::Type termType, VolumeField<scalar>& rho, VolumeField<ValueType>& field
    );

    ~DdtOperator();

    void explicitOperation(Vector<ValueType>& source, scalar t, scalar dt) const;

    // Format-generic implicit temporal assembly. Defined here (not ddtOperator.cpp) so any TU
    // can instantiate it for any SystemMatrixType without needing an explicit instantiation --
    // same reasoning as SourceTerm::implicitOperation<SystemMatrixType>. SystemMatrixType is
    // deduced from ls's own type, so the existing CSR call site (TemporalOperator's type
    // erasure) is unaffected.
    template<typename SystemMatrixType>
    void implicitOperation(
        la::LinearSystem<ValueType, ValueType, SystemMatrixType>& ls, scalar t, scalar dt
    ) const
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

    template<typename SystemMatrixType>
    void bdf1Kernel(la::LinearSystem<ValueType, ValueType, SystemMatrixType>& ls, scalar, scalar dt)
        const
    {
        const auto vol = this->getVector().mesh().cellVolumes().view();
        const auto operatorScaling = this->getCoefficient();
        const auto oldVector = oldTime(this->field_).internalVector().view();
        auto [rhs, values] = views(ls.rhs(), ls.matrix().values());
        const auto ma = ls.matrix().faceToMatrixView();

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
                    values[ma.diagIdx(celli)] +=
                        rhoNew[celli] * commonCoef * a0a1 * one<ValueType>();
                    rhs[celli] += rhoOld[celli] * commonCoef * a0a1 * oldVector[celli];
                },
                "ddtOperator::implicitOperation<BDF1,rho>"
            );
        }
    }

    template<typename SystemMatrixType>
    void bdf2Kernel(la::LinearSystem<ValueType, ValueType, SystemMatrixType>& ls, scalar, scalar dt)
        const
    {
        const auto vol = this->getVector().mesh().cellVolumes().view();
        const auto operatorScaling = this->getCoefficient();
        auto& old = oldTime(this->field_);
        auto& oldOld = oldTime(old);
        const auto [oldVector, oldOldVector] = views(old.internalVector(), oldOld.internalVector());
        auto [rhs, values] = views(ls.rhs(), ls.matrix().values());

        const auto ma = ls.matrix().faceToMatrixView();

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

    // Segregated counterpart of bdf1Kernel above -- same structure, but the diagonal coefficient
    // is a scalar shared across every rhs component instead of a ValueType. Header-inline and
    // SystemMatrixType-generic for the same reason as bdf1Kernel.
    template<typename SystemMatrixType>
    void bdf1KernelScalarMtx(
        la::LinearSystem<scalar, ValueType, SystemMatrixType>& ls, scalar, scalar dt
    ) const
    {
        const auto vol = this->getVector().mesh().cellVolumes().view();
        const auto operatorScaling = this->getCoefficient();
        const auto oldVector = oldTime(this->field_).internalVector().view();
        auto [rhs, values] = views(ls.rhs(), ls.matrix().values());
        const auto ma = ls.matrix().faceToMatrixView();

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

    template<typename SystemMatrixType>
    void bdf2KernelScalarMtx(
        la::LinearSystem<scalar, ValueType, SystemMatrixType>& ls, scalar, scalar dt
    ) const
    {
        const auto vol = this->getVector().mesh().cellVolumes().view();
        const auto operatorScaling = this->getCoefficient();
        auto& old = oldTime(this->field_);
        auto& oldOld = oldTime(old);
        const auto [oldVector, oldOldVector] = views(old.internalVector(), oldOld.internalVector());
        auto [rhs, values] = views(ls.rhs(), ls.matrix().values());
        const auto ma = ls.matrix().faceToMatrixView();

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

    /* @brief Implicit temporal assembly into a scalar-matrix / ValueType-rhs linear system
     *        (segregated vector-solve form). Only present when ValueType != scalar; for scalar
     *        fields the same-type overload above already covers LinearSystem<scalar, scalar>.
     *        The scalar diagonal entry scales every rhs component equally. Format-generic over
     *        SystemMatrixType (CSR or ELL) and header-inline, same reasoning as
     *        implicitOperation<SystemMatrixType> above. Guarded on ValueType (the class's own
     *        parameter, not a separate member-template one), so unlike the old declare-in-header
     *        /define-in-cpp version, SystemMatrixType is the only substitutable parameter and is
     *        always deduced from ls -- no risk of an explicit template argument silently picking
     *        an unrelated, never-defined specialization.
     */
    template<typename SystemMatrixType>
        requires(!std::is_same_v<ValueType, scalar>)
    void implicitOperation(
        la::LinearSystem<scalar, ValueType, SystemMatrixType>& ls, scalar t, scalar dt
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

    DdtScheme scheme() const noexcept { return scheme_; }

    void read(const Input&);

    std::string getName() const { return "DdtOperator"; }

private:

    // NOTE ddtOperator does not have a FactoryClass

    DdtScheme scheme_ {DdtScheme::BDF1};

    // Non-null → density-weighted form ddt(rho, field). Null → stock single-coefficient form.
    VolumeField<scalar>* rho_ {nullptr};
};


} // namespace NeoN
