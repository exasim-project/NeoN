// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/fields/field.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/vector/vectorTypeDefs.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/upwind.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenGrad.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/tensorVecField.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/finiteVolume/cellCentred/stencil/geometryScheme.hpp"

#include <Kokkos_Core.hpp>

#include <functional>

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
void computeLinearUpwindInterpolation(
    const VolumeField<ValueType>& src,
    const SurfaceField<scalar>& flux,
    const VolumeField<Vec3>& gradPhi,
    const vectorVector& faceCentres,
    const vectorVector& cellCentres,
    const UnstructuredMesh& mesh,
    SurfaceField<ValueType>& dst
);

void computeLinearUpwindInterpolation(
    const VolumeField<Vec3>& src,
    const SurfaceField<scalar>& flux,
    const TensorVecField& gradU,
    const vectorVector& faceCentres,
    const vectorVector& cellCentres,
    const UnstructuredMesh& mesh,
    SurfaceField<Vec3>& dst
);

template<typename ValueType>
class LinearUpwind :
    public SurfaceInterpolationFactory<ValueType>::template Register<LinearUpwind<ValueType>>
{

    using Base =
        SurfaceInterpolationFactory<ValueType>::template Register<LinearUpwind<ValueType>>;

public:

    LinearUpwind(const Executor& exec, const UnstructuredMesh& mesh, [[maybe_unused]] Input input)
        : Base(exec, mesh), faceCentres_(mesh.faceCentres()), cellCentres_(mesh.cellCentres()),
          gaussGreenGrad_(exec, mesh),
          geometryScheme_(GeometryScheme::readOrCreate(mesh)) {};

    static std::string name() { return "linearUpwind"; }

    static std::string doc() { return "linearUpwind interpolation"; }

    static std::string schema() { return "none"; }

    void interpolate(
        [[maybe_unused]] const VolumeField<ValueType>& src,
        [[maybe_unused]] SurfaceField<ValueType>& dst
    ) const override
    {
        NF_ERROR_EXIT("linearUpwind scheme requires a faceFlux");
    }

    void interpolate(
        const SurfaceField<scalar>& flux,
        const VolumeField<ValueType>& src,
        SurfaceField<ValueType>& dst
    ) const override;

    void weight(const VolumeField<ValueType>&, SurfaceField<scalar>&) const override
    {
        NF_ERROR_EXIT("linearUpwind interpolation scheme requires a faceFlux");
    }

    void weight(
        const SurfaceField<scalar>& faceFlux,
        const VolumeField<ValueType>& src,
        SurfaceField<scalar>& weights
    ) const override
    {
        computeUpwindInterpolationWeights(faceFlux, src, weights);
    }

    std::unique_ptr<SurfaceInterpolationFactory<ValueType>> clone() const override
    {
        return std::make_unique<LinearUpwind>(*this);
    }

private:

    vectorVector faceCentres_;
    vectorVector cellCentres_;
    GaussGreenGrad gaussGreenGrad_;
    const std::shared_ptr<GeometryScheme> geometryScheme_;
};

} // namespace NeoN


namespace NeoN
{

namespace fvcc = finiteVolume::cellCentred;

template class fvcc::LinearUpwind<scalar>;
template class fvcc::LinearUpwind<Vec3>;

}
