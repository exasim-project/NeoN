// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/finiteVolume/cellCentred/faceNormalGradient/faceNormalGradient.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/stencil/geometryScheme.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/gaussGreenGrad.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/linear.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

#include <Kokkos_Core.hpp>

#include <functional>


namespace NeoN::finiteVolume::cellCentred
{

void computeCorrectedFaceNormalGrad(
    const VolumeField<scalar>& volField,
    const std::shared_ptr<GeometryScheme> geometryScheme,
    const GaussGreenGrad& grad,
    const SurfaceInterpolation<Vec3>& surfInterpVec3,
    SurfaceField<scalar>& surfaceField
);

void computeCorrectedFaceNormalGrad(
    const VolumeField<Vec3>& volField,
    const std::shared_ptr<GeometryScheme> geometryScheme,
    const GaussGreenGrad& grad,
    const SurfaceInterpolation<Vec3>& surfInterpVec3,
    SurfaceField<Vec3>& surfaceField
);

void computeCorrection(
    const VolumeField<scalar>& volField,
    const std::shared_ptr<GeometryScheme> geometryScheme,
    const GaussGreenGrad& grad,
    const SurfaceInterpolation<Vec3>& surfInterpVec3,
    SurfaceField<scalar>& correctionField
);

void computeCorrection(
    const VolumeField<Vec3>& volField,
    const std::shared_ptr<GeometryScheme> geometryScheme,
    const GaussGreenGrad& grad,
    const SurfaceInterpolation<Vec3>& surfInterpVec3,
    SurfaceField<Vec3>& correctionField
);

template<typename ValueType>
class Corrected :
    public FaceNormalGradientFactory<ValueType>::template Register<Corrected<ValueType>>
{
    using Base = FaceNormalGradientFactory<ValueType>::template Register<Corrected<ValueType>>;


public:

    Corrected(const Executor& exec, const UnstructuredMesh& mesh, Input)
        : Base(exec, mesh), geometryScheme_(GeometryScheme::readOrCreate(mesh)),
          grad_(exec, mesh),
          surfInterpVec3_(
              exec, mesh, std::make_unique<Linear<Vec3>>(exec, mesh, Dictionary())
          ) {};

    Corrected(const Executor& exec, const UnstructuredMesh& mesh)
        : Base(exec, mesh), geometryScheme_(GeometryScheme::readOrCreate(mesh)),
          grad_(exec, mesh),
          surfInterpVec3_(
              exec, mesh, std::make_unique<Linear<Vec3>>(exec, mesh, Dictionary())
          ) {};

    static std::string name() { return "corrected"; }

    static std::string doc() { return "Corrected interpolation with non-orthogonal correction"; }

    static std::string schema() { return "none"; }

    virtual void faceNormalGrad(
        const VolumeField<ValueType>& volField, SurfaceField<ValueType>& surfaceField
    ) const override
    {
        computeCorrectedFaceNormalGrad(volField, geometryScheme_, grad_, surfInterpVec3_, surfaceField);
    }

    virtual bool corrected() const override { return true; }

    virtual void correction(
        const VolumeField<ValueType>& volField, SurfaceField<ValueType>& correctionField
    ) const override
    {
        computeCorrection(volField, geometryScheme_, grad_, surfInterpVec3_, correctionField);
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
    GaussGreenGrad grad_;
    SurfaceInterpolation<Vec3> surfInterpVec3_;
};

// instantiate the template class
template class Corrected<scalar>;
template class Corrected<Vec3>;

} // namespace NeoN
