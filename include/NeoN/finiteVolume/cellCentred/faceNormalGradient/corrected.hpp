// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/faceNormalGradient.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenGrad.hpp"
#include "NeoN/finiteVolume/cellCentred/stencil/geometryScheme.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

#include <Kokkos_Core.hpp>

#include <functional>


namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
void computeCorrectedFaceNormalGrad(
    const VolumeField<ValueType>& volField,
    const std::shared_ptr<GeometryScheme> geometryScheme,
    SurfaceField<ValueType>& surfaceField
);

template<typename ValueType>
class Corrected :
    public FaceNormalGradientFactory<ValueType>::template Register<Corrected<ValueType>>
{
    using Base = FaceNormalGradientFactory<ValueType>::template Register<Corrected<ValueType>>;

public:

    Corrected(const Executor& exec, const UnstructuredMesh& mesh, Input)
        : Base(exec, mesh), geometryScheme_(GeometryScheme::readOrCreate(mesh)) {};

    Corrected(const Executor& exec, const UnstructuredMesh& mesh)
        : Base(exec, mesh), geometryScheme_(GeometryScheme::readOrCreate(mesh)) {};

    static std::string name() { return "corrected"; }

    static std::string doc()
    {
        return "Corrected face normal gradient with explicit non-orthogonality correction";
    }

    static std::string schema() { return "none"; }

    virtual void faceNormalGrad(
        const VolumeField<ValueType>& volField, SurfaceField<ValueType>& surfaceField
    ) const override
    {
        computeCorrectedFaceNormalGrad(volField, geometryScheme_, surfaceField);
    }

    virtual const SurfaceField<scalar>& deltaCoeffs() const override
    {
        return geometryScheme_->nonOrthDeltaCoeffs();
    }

    std::unique_ptr<FaceNormalGradientFactory<ValueType>> clone() const override
    {
        return std::make_unique<Corrected>(*this);
    }

private:

    const std::shared_ptr<GeometryScheme> geometryScheme_;
};

// instantiate the template class
template class Corrected<scalar>;
template class Corrected<Vec3>;

} // namespace NeoN
