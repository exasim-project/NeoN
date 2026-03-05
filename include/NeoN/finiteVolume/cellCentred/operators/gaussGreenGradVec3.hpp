// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
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

class GaussGreenGradVec3 :
    public GradOperatorFactory<Tensor>::template Register<GaussGreenGradVec3>
{

    using Base = GradOperatorFactory<Tensor>::template Register<GaussGreenGradVec3>;

public:

    static std::string name() { return "Gauss"; }

    static std::string doc() { return "Gauss-Green Gradient for Vec3 fields"; }

    static std::string schema() { return "none"; }

    GaussGreenGradVec3(const Executor& exec, const UnstructuredMesh& mesh);

    virtual void
    grad(const VolumeField<Vec3>&, const dsl::Coeff, la::LinearSystem<Tensor>&) const override
    {
        NF_ERROR_EXIT("Not implemented");
    };

    virtual void grad(
        const VolumeField<Vec3>& phi, const dsl::Coeff operatorScaling, Vector<Tensor>& gradPhi
    ) const override;

    VolumeField<Tensor> grad(
        const VolumeField<Vec3>& phi, const dsl::Coeff operatorScaling = dsl::Coeff {}
    ) const override;

    virtual std::unique_ptr<GradOperatorFactory<Tensor>> clone() const override
    {
        NF_ERROR_EXIT("Not implemented");
    };

private:

    SurfaceInterpolation<Vec3> surfaceInterpolation_;
};

} // namespace NeoN
