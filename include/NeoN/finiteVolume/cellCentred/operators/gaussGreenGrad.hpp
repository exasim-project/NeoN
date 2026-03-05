// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gradOperator.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"

namespace NeoN::finiteVolume::cellCentred
{

class GaussGreenGrad : public GradOperatorFactory<Vec3>::template Register<GaussGreenGrad>
{

    using Base = GradOperatorFactory<Vec3>::template Register<GaussGreenGrad>;

public:

    static std::string name() { return "Gauss"; }

    static std::string doc() { return "Gauss-Green Gradient"; }

    static std::string schema() { return "none"; }

    GaussGreenGrad(const Executor& exec, const UnstructuredMesh& mesh);

    virtual void
    grad(const VolumeField<scalar>&, const dsl::Coeff, la::LinearSystem<Vec3>&) const override
    {
        NF_ERROR_EXIT("Not implemented");
    };

    virtual void grad(
        const VolumeField<scalar>& phi, const dsl::Coeff operatorScaling, Vector<Vec3>& gradPhi
    ) const override;

    virtual void
    grad(const VolumeField<scalar>& phi, const dsl::Coeff coeff, VolumeField<Vec3>& in) const
    {
        grad(phi, coeff, in.internalVector());
    }

    VolumeField<Vec3> grad(
        const VolumeField<scalar>& phi, const dsl::Coeff operatorScaling = dsl::Coeff {}
    ) const override;

    virtual std::unique_ptr<GradOperatorFactory<Vec3>> clone() const override
    {
        NF_ERROR_EXIT("Not implemented");
    };

private:

    SurfaceInterpolation<scalar> surfaceInterpolation_;
};

} // namespace NeoN
