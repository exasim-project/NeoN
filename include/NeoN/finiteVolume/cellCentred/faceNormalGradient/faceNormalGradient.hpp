// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#pragma once

#include <functional>

#include <Kokkos_Core.hpp>

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/input.hpp"
#include "NeoN/core/runtimeSelectionFactory.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
class FaceNormalGradientFactory :
    public NeoN::RuntimeSelectionFactory<
        FaceNormalGradientFactory<ValueType>,
        Parameters<const Executor&, const UnstructuredMesh&, const Input&>>
{

public:

    static std::unique_ptr<FaceNormalGradientFactory>
    create(const Executor& exec, const UnstructuredMesh& uMesh, const Input& inputs)
    {
        // input is dictionary the key is "interpolation"
        std::string key =
            (std::holds_alternative<NeoN::Dictionary>(inputs))
                ? std::get<NeoN::Dictionary>(inputs).get<std::string>("faceNormalGradient")
                : std::get<NeoN::TokenList>(inputs).next<std::string>();

        FaceNormalGradientFactory<ValueType>::keyExistsOrError(key);
        return FaceNormalGradientFactory<ValueType>::table().at(key)(exec, uMesh, inputs);
    }

    static std::string name() { return "FaceNormalGradientFactory"; }

    FaceNormalGradientFactory(const Executor& exec, const UnstructuredMesh& mesh)
        : exec_(exec), mesh_(mesh) {};

    virtual ~FaceNormalGradientFactory() {} // Virtual destructor

    virtual void faceNormalGrad(
        const VolumeField<ValueType>& volField, SurfaceField<ValueType>& surfaceField
    ) const = 0;

    virtual const SurfaceField<scalar>& deltaCoeffs() const = 0;

    // Pure virtual function for cloning
    virtual std::unique_ptr<FaceNormalGradientFactory<ValueType>> clone() const = 0;

protected:

    const Executor exec_;
    const UnstructuredMesh& mesh_;
};

template<typename ValueType>
class FaceNormalGradient
{

public:

    FaceNormalGradient(const FaceNormalGradient& faceNGrad)
        : exec_(faceNGrad.exec_), mesh_(faceNGrad.mesh_),
          faceNormalGradKernel_(faceNGrad.faceNormalGradKernel_->clone()) {};

    FaceNormalGradient(FaceNormalGradient&& faceNGrad)
        : exec_(faceNGrad.exec_), mesh_(faceNGrad.mesh_),
          faceNormalGradKernel_(std::move(faceNGrad.faceNormalGradKernel_)) {};

    FaceNormalGradient(
        const Executor& exec,
        const UnstructuredMesh& mesh,
        std::unique_ptr<FaceNormalGradientFactory<ValueType>> faceNormalGradient
    )
        : exec_(exec), mesh_(mesh), faceNormalGradKernel_(std::move(faceNormalGradient)) {};

    FaceNormalGradient(const Executor& exec, const UnstructuredMesh& mesh, const Input& input)
        : exec_(exec), mesh_(mesh),
          faceNormalGradKernel_(FaceNormalGradientFactory<ValueType>::create(exec, mesh, input)) {};


    void faceNormalGrad(
        const VolumeField<ValueType>& volField, SurfaceField<ValueType>& surfaceField
    ) const
    {
        faceNormalGradKernel_->faceNormalGrad(volField, surfaceField);
    }

    const SurfaceField<scalar>& deltaCoeffs() const { return faceNormalGradKernel_->deltaCoeffs(); }


    SurfaceField<ValueType> faceNormalGrad(const VolumeField<ValueType>& volField) const
    {
        std::string nameInterpolated = "interpolated_" + volField.name;
        SurfaceField<ValueType> surfaceField(
            exec_, nameInterpolated, mesh_, createCalculatedBCs<SurfaceBoundary<ValueType>>(mesh_)
        );
        faceNormalGrad(volField, surfaceField);
        return surfaceField;
    }

private:

    const Executor exec_;
    const UnstructuredMesh& mesh_;
    std::unique_ptr<FaceNormalGradientFactory<ValueType>> faceNormalGradKernel_;
};


} // namespace NeoN
