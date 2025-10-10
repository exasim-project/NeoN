// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
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

    // fvcc::VolumeField<Vec3> grad(const fvcc::VolumeField<scalar>& phi);

    // virtual void grad(const VolumeField<scalar>& phi, VolumeField<Vec3>& gradPhi) const;

    virtual VolumeField<Vec3> grad(const VolumeField<scalar>& phi);

    virtual void grad(
        la::LinearSystem<Vec3, localIdx>& ls,
        const VolumeField<scalar>& phi,
        const dsl::Coeff operatorScaling
    ) const override
    {
        NF_ERROR_EXIT("Not implemented");
    };

    virtual void grad(const VolumeField<scalar>& phi, Vector<Vec3>& gradPhi) const;

    virtual void grad(
        VolumeField<Vec3>& gradPhi, const VolumeField<scalar>& phi, const dsl::Coeff operatorScaling
    ) const
    {
        NF_ERROR_EXIT("Not implemented");
    };

    virtual VolumeField<Vec3>
    grad(const VolumeField<scalar>& phi, const dsl::Coeff operatorScaling) const
    {
        NF_ERROR_EXIT("Not implemented");
    }

    virtual std::unique_ptr<GradOperatorFactory<Vec3>> clone() const
    {
        NF_ERROR_EXIT("Not implemented");
    };

private:

    SurfaceInterpolation<scalar> surfaceInterpolation_;
};

} // namespace NeoN
