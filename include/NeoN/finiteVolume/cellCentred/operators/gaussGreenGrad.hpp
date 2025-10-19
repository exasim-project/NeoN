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

    /* @brief compute gradient for implicit operator
     *
     * @param ls [in,out] - assemble gradient operator into the given linear system
     * @param phi [in] -
     * @param operatorScaling [in] -
     */
    virtual void grad(
        la::LinearSystem<Vec3, localIdx>&, const VolumeField<scalar>&, const dsl::Coeff
    ) const override
    {
        NF_ERROR_EXIT("Not implemented");
    };

    virtual void grad(
        const VolumeField<scalar>& phi, const dsl::Coeff operatorScaling, Vector<Vec3>& gradPhi
    ) const;

    /* @brief compute grad
     *
     * @param phi [in]
     * @param operatorScaling [in]
     * @param gradPhi [in,out] - resulting gradient field
     */
    virtual void grad(const VolumeField<scalar>&, const dsl::Coeff, VolumeField<Vec3>&) const
    {
        NF_ERROR_EXIT("Not implemented");
    };

    VolumeField<Vec3>
    grad(const VolumeField<scalar>& phi, const dsl::Coeff operatorScaling = dsl::Coeff {}) const;

    virtual std::unique_ptr<GradOperatorFactory<Vec3>> clone() const
    {
        NF_ERROR_EXIT("Not implemented");
    };

private:

    SurfaceInterpolation<scalar> surfaceInterpolation_;
};

} // namespace NeoN
