// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#pragma once

#include "NeoFOAM/core/executor/executor.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/faceNormalGradient/faceNormalGradient.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoFOAM/finiteVolume/cellCentred/stencil/geometryScheme.hpp"
#include "NeoFOAM/mesh/unstructured/unstructuredMesh.hpp"

#include <Kokkos_Core.hpp>

#include <functional>


namespace NeoFOAM::finiteVolume::cellCentred
{

template<typename ValueType>
void computeFaceNormalGrad(
    const VolumeField<ValueType>& volField,
    const std::shared_ptr<GeometryScheme> geometryScheme,
    SurfaceField<ValueType>& surfaceField
);

template<typename ValueType>
class Uncorrected :
    public FaceNormalGradientFactory<ValueType>::template Register<Uncorrected<ValueType>>
{
    using Base = FaceNormalGradientFactory<ValueType>::template Register<Uncorrected<ValueType>>;


public:

    Uncorrected(const Executor& exec, const UnstructuredMesh& mesh, Input)
        : Base(exec, mesh), geometryScheme_(GeometryScheme::readOrCreate(mesh)) {};

    Uncorrected(const Executor& exec, const UnstructuredMesh& mesh)
        : Base(exec, mesh), geometryScheme_(GeometryScheme::readOrCreate(mesh)) {};

    static std::string name() { return "uncorrected"; }

    static std::string doc() { return "Uncorrected interpolation"; }

    static std::string schema() { return "none"; }

    virtual void faceNormalGrad(
        const VolumeField<ValueType>& volField, SurfaceField<ValueType>& surfaceField
    ) const override
    {
        computeFaceNormalGrad(volField, geometryScheme_, surfaceField);
    }

    virtual const SurfaceField<scalar>& deltaCoeffs() const override
    {
        return geometryScheme_->nonOrthDeltaCoeffs();
    }

    std::unique_ptr<FaceNormalGradientFactory<ValueType>> clone() const override
    {
        return std::make_unique<Uncorrected>(*this);
    }

private:

    const std::shared_ptr<GeometryScheme> geometryScheme_;
};

// instantiate the template class
template class Uncorrected<scalar>;
template class Uncorrected<Vector>;

} // namespace NeoFOAM
