// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/faceNormalGradient.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/stencil/geometryScheme.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

#include <Kokkos_Core.hpp>

#include <functional>


namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
void computeFaceNormalGrad(
    const VolumeField<ValueType>& volVector,
    const std::shared_ptr<GeometryScheme> geometryScheme,
    SurfaceField<ValueType>& surfaceVector
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
        const VolumeField<ValueType>& volVector, SurfaceField<ValueType>& surfaceVector
    ) const override
    {
        computeFaceNormalGrad(volVector, geometryScheme_, surfaceVector);
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
template class Uncorrected<Vec3>;

} // namespace NeoN
