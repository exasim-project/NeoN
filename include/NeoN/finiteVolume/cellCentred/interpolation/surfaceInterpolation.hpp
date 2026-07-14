// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>

#include <Kokkos_Core.hpp>

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/input.hpp"
#include "NeoN/core/runtimeSelectionFactory.hpp"
#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/inlineInterpolationKernels.hpp"

namespace NeoN::finiteVolume::cellCentred
{

/* @class SurfaceInterpolationFactory
**
*/
template<typename ValueType>
class SurfaceInterpolationFactory :
    public RuntimeSelectionFactory<
        SurfaceInterpolationFactory<ValueType>,
        Parameters<const Executor&, const UnstructuredMesh&, const Input&>>
{
    using ScalarSurfaceField = SurfaceField<scalar>;

public:

    static std::unique_ptr<SurfaceInterpolationFactory<ValueType>>
    create(const Executor& exec, const UnstructuredMesh& uMesh, const Input& inputs)
    {
        // input is dictionary the key is "interpolation"
        std::string key =
            (std::holds_alternative<NeoN::Dictionary>(inputs))
                ? std::get<NeoN::Dictionary>(inputs).get<std::string>("surfaceInterpolation")
                : std::get<NeoN::TokenList>(inputs).next<std::string>();

        SurfaceInterpolationFactory<ValueType>::keyExistsOrError(key);
        return SurfaceInterpolationFactory<ValueType>::table().at(key)(exec, uMesh, inputs);
    }

    static std::string name() { return "SurfaceInterpolationFactory"; }

    SurfaceInterpolationFactory(const Executor& exec, const UnstructuredMesh& mesh)
        : exec_(exec), mesh_(mesh) {};

    virtual ~SurfaceInterpolationFactory() {} // Virtual destructor

    virtual void
    interpolate(const VolumeField<ValueType>& src, SurfaceField<ValueType>& dst) const = 0;

    virtual void interpolate(
        const SurfaceField<scalar>& flux,
        const VolumeField<ValueType>& src,
        SurfaceField<ValueType>& dst
    ) const = 0;

    virtual void weight(const VolumeField<ValueType>& src, SurfaceField<scalar>& weight) const = 0;

    virtual void weight(
        const SurfaceField<scalar>& flux,
        const VolumeField<ValueType>& src,
        SurfaceField<scalar>& weight
    ) const = 0;

    /* @brief Whether this scheme adds an explicit (deferred) correction on top of its implicit
     * weights, e.g. linearUpwind. For corrected schemes interpolate() returns the weighted value
     * plus correction(); implicit assemblers add surfaceIntegrate(faceFlux*correction()) to the
     * rhs.
     */
    virtual bool corrected() const { return false; }

    /* @brief The explicit correction part of interpolate(), i.e. interpolate() minus the value
     * reconstructed from weight(). Defaults to zero for uncorrected schemes; assemblers only call
     * it when corrected() returns true.
     */
    virtual void correction(
        [[maybe_unused]] const SurfaceField<scalar>& flux,
        [[maybe_unused]] const VolumeField<ValueType>& src,
        SurfaceField<ValueType>& corr
    ) const
    {
        fill(corr.internalVector(), zero<ValueType>());
        fill(corr.boundaryData().value(), zero<ValueType>());
    }

    /* @brief Returns a device-callable weight kernel for use inside Kokkos kernels.
     * The kernel computes the face interpolation weight inline per face without virtual dispatch.
     */
    virtual InlineWeightKernel inlineWeightKernel() const = 0;

    // Pure virtual function for cloning
    virtual std::unique_ptr<SurfaceInterpolationFactory<ValueType>> clone() const = 0;

protected:

    const Executor exec_;
    const UnstructuredMesh& mesh_;
};

template<typename ValueType>
class SurfaceInterpolation
{

    using VectorValueType = ValueType;

public:

    SurfaceInterpolation(const SurfaceInterpolation& surfInterp)
        : exec_(surfInterp.exec_), mesh_(surfInterp.mesh_),
          interpolationKernel_(surfInterp.interpolationKernel_->clone()) {};

    SurfaceInterpolation(SurfaceInterpolation&& surfInterp)
        : exec_(surfInterp.exec_), mesh_(surfInterp.mesh_),
          interpolationKernel_(std::move(surfInterp.interpolationKernel_)) {};

    SurfaceInterpolation(
        const Executor& exec,
        const UnstructuredMesh& mesh,
        std::unique_ptr<SurfaceInterpolationFactory<ValueType>> interpolationKernel
    )
        : exec_(exec), mesh_(mesh), interpolationKernel_(std::move(interpolationKernel)) {};

    SurfaceInterpolation(const Executor& exec, const UnstructuredMesh& mesh, const Input& input)
        : exec_(exec), mesh_(mesh),
          interpolationKernel_(SurfaceInterpolationFactory<ValueType>::create(exec, mesh, input)) {
          };


    void interpolate(const VolumeField<ValueType>& src, SurfaceField<ValueType>& dst) const
    {
        interpolationKernel_->interpolate(src, dst);
    }

    void interpolate(
        const SurfaceField<scalar>& flux,
        const VolumeField<ValueType>& src,
        SurfaceField<ValueType>& dst
    ) const
    {
        interpolationKernel_->interpolate(flux, src, dst);
    }

    void weight(const VolumeField<ValueType>& src, SurfaceField<scalar>& weight) const
    {
        interpolationKernel_->weight(src, weight);
    }

    void weight(
        const SurfaceField<scalar>& flux,
        const VolumeField<ValueType>& src,
        SurfaceField<scalar>& weight
    ) const
    {
        interpolationKernel_->weight(flux, src, weight);
    }

    bool corrected() const { return interpolationKernel_->corrected(); }

    InlineWeightKernel inlineWeightKernel() const
    {
        return interpolationKernel_->inlineWeightKernel();
    }

    void correction(
        const SurfaceField<scalar>& flux,
        const VolumeField<ValueType>& src,
        SurfaceField<ValueType>& corr
    ) const
    {
        interpolationKernel_->correction(flux, src, corr);
    }


    SurfaceField<ValueType> interpolate(const VolumeField<ValueType>& src) const
    {
        std::string nameInterpolated = "interpolated_" + src.name;
        SurfaceField<ValueType> dst(
            exec_, nameInterpolated, mesh_, createCalculatedBCs<SurfaceBoundary<ValueType>>(mesh_)
        );
        interpolate(src, dst);
        return dst;
    }

    SurfaceField<ValueType>
    interpolate(const SurfaceField<ValueType>& flux, const VolumeField<ValueType>& src) const
    {
        std::string name = "interpolated_" + src.name;
        SurfaceField<ValueType> dst(
            exec_, name, mesh_, createCalculatedBCs<SurfaceBoundary<ValueType>>(mesh_)
        );
        interpolate(flux, src, dst);
        return dst;
    }

    SurfaceField<scalar> weight(const VolumeField<ValueType>& src) const
    {
        std::string name = "weight_" + src.name;
        SurfaceField<scalar> weightVector(
            exec_, name, mesh_, createCalculatedBCs<SurfaceBoundary<scalar>>(mesh_)
        );
        weight(src, weightVector);
        return weightVector;
    }

    SurfaceField<scalar>
    weight(const SurfaceField<scalar>& flux, const VolumeField<ValueType>& src) const
    {
        std::string name = "weight_" + src.name;
        SurfaceField<scalar> weightVector(
            exec_, name, mesh_, createCalculatedBCs<SurfaceBoundary<scalar>>(mesh_)
        );
        weight(flux, src, weightVector);
        return weightVector;
    }

private:

    const Executor exec_;
    const UnstructuredMesh& mesh_;
    std::unique_ptr<SurfaceInterpolationFactory<ValueType>> interpolationKernel_;
};


} // namespace NeoN
