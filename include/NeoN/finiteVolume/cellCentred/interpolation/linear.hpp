// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#pragma once

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/stencil/geometryScheme.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

#include <Kokkos_Core.hpp>

#include <functional>


namespace NeoN::finiteVolume::cellCentred
{

/* @brief computional kernel to perform a linear interpolation
** from a source volumeField to a surface field. It performs an interpolation
** of the form
**
** d_f = w_f * s_O + ( 1 - w_f ) * s_N
**
**@param src the input field
**@param weights weights for the interpolation
**@param dst the target field
*/
template<typename ValueType>
void computeLinearInterpolation(
    const VolumeField<ValueType>& src,
    const SurfaceField<scalar>& weights,
    SurfaceField<ValueType>& dst
);

template<typename ValueType>
class Linear : public SurfaceInterpolationFactory<ValueType>::template Register<Linear<ValueType>>
{
    using Base = SurfaceInterpolationFactory<ValueType>::template Register<Linear<ValueType>>;

public:

    Linear(const Executor& exec, const UnstructuredMesh& mesh, [[maybe_unused]] Input input)
        : Base(exec, mesh), geometryScheme_(GeometryScheme::readOrCreate(mesh)) {};

    Linear(const Executor& exec, const UnstructuredMesh& mesh)
        : Base(exec, mesh), geometryScheme_(GeometryScheme::readOrCreate(mesh)) {};


    static std::string name() { return "linear"; }

    static std::string doc() { return "linear interpolation"; }

    static std::string schema() { return "none"; }

    void interpolate(const VolumeField<ValueType>& src, SurfaceField<ValueType>& dst) const override
    {
        computeLinearInterpolation(src, geometryScheme_->weights(), dst);
    }

    void interpolate(
        const SurfaceField<scalar>&, const VolumeField<ValueType>& src, SurfaceField<ValueType>& dst
    ) const override
    {
        interpolate(src, dst);
    }

    void weight(const VolumeField<ValueType>& src, SurfaceField<scalar>& weight) const override
    {
        const SurfaceField<scalar>& linearWeight = geometryScheme_->weights();
        weight.internalField() = linearWeight.internalField();
        weight.boundaryField() = linearWeight.boundaryField();
    }

    void weight(
        const SurfaceField<scalar>& flux,
        const VolumeField<ValueType>& src,
        SurfaceField<scalar>& weight
    ) const override
    {
        const SurfaceField<scalar>& linearWeight = geometryScheme_->weights();
        weight.internalField() = linearWeight.internalField();
        weight.boundaryField() = linearWeight.boundaryField();
    }


    std::unique_ptr<SurfaceInterpolationFactory<ValueType>> clone() const override
    {
        return std::make_unique<Linear>(*this);
    }

private:

    const std::shared_ptr<GeometryScheme> geometryScheme_;
};

} // namespace NeoN

namespace NeoN
{

namespace fvcc = finiteVolume::cellCentred;

template class fvcc::SurfaceInterpolationFactory<scalar>;
template class fvcc::SurfaceInterpolationFactory<Vec3>;

template class fvcc::Linear<scalar>;
template class fvcc::Linear<Vec3>;

}
