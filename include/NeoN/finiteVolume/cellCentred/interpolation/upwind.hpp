// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#pragma once

#include "NeoN/fields/field.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/finiteVolume/cellCentred/stencil/geometryScheme.hpp"

#include <Kokkos_Core.hpp>

#include <functional>

namespace NeoN::finiteVolume::cellCentred
{

/* @brief computional kernel to perform an upwind interpolation
** from a source volumeVector to a surface field. It performs an interpolation
** of the form
**
** d_f = w_f * s_O + ( 1 - w_f ) * s_N where w_1 is either 0,1 depending on the
** direction of the flux
**
**@param src the input field
**@param weights weights for the interpolation
**@param dst the target field
*/
template<typename ValueType>
void computeUpwindInterpolation(
    const VolumeField<ValueType>& src,
    const SurfaceField<scalar>& flux,
    const SurfaceField<scalar>& weights,
    SurfaceField<ValueType>& dst
);

template<typename ValueType>
void computeUpwindInterpolationFace(
    const VolumeField<ValueType>& src,
    const SurfaceField<scalar>& flux,
    const SurfaceField<scalar>& weights,
    SurfaceField<ValueType>& dst,
    const localIdx faceIdx
);

template<typename ValueType>
class Upwind : public SurfaceInterpolationFactory<ValueType>::template Register<Upwind<ValueType>>
{

    using Base = SurfaceInterpolationFactory<ValueType>::template Register<Upwind<ValueType>>;

public:

    Upwind(const Executor& exec, const UnstructuredMesh& mesh, [[maybe_unused]] Input input)
        : Base(exec, mesh), geometryScheme_(GeometryScheme::readOrCreate(mesh)) {};

    static std::string name() { return "upwind"; }

    static std::string doc() { return "upwind interpolation"; }

    static std::string schema() { return "none"; }

    void interpolate(
        [[maybe_unused]] const VolumeField<ValueType>& src,
        [[maybe_unused]] SurfaceField<ValueType>& dst
    ) const override
    {
        NF_ERROR_EXIT("limited scheme requires a faceFlux");
    }

    void interpolate(
        const SurfaceField<scalar>& flux,
        const VolumeField<ValueType>& src,
        SurfaceField<ValueType>& dst
    ) const override
    {
        computeUpwindInterpolation(src, flux, geometryScheme_->weights(), dst);
    }

    // src weight
    void weight(const VolumeField<ValueType>&, SurfaceField<scalar>&) const override
    {
        NF_ERROR_EXIT("limited scheme requires a faceFlux");
    }

    // flux, src, weight
    void weight(const SurfaceField<scalar>&, const VolumeField<ValueType>&, SurfaceField<scalar>&)
        const override
    {
        // TODO should that be implemented?
        // geometryScheme_->weights(flux, src, weight);
    }

    std::unique_ptr<SurfaceInterpolationFactory<ValueType>> clone() const override
    {
        return std::make_unique<Upwind>(*this);
    }

private:

    const std::shared_ptr<GeometryScheme> geometryScheme_;
};

} // namespace NeoN


namespace NeoN
{

namespace fvcc = finiteVolume::cellCentred;

template class fvcc::Upwind<scalar>;
template class fvcc::Upwind<Vec3>;

}
