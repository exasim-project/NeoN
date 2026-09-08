// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/linear.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoN/finiteVolume/cellCentred/stencil/geometryScheme.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

#include <Kokkos_Core.hpp>

namespace NeoN::finiteVolume::cellCentred
{

/* @brief blended central/upwind weights of the limitedLinear scheme.
 *
 * w_f = limiter * w_CD + (1 - limiter) * w_upwind, with the limiter built from the TVD ratio r of
 * the upwind-cell gradient to the face gradient: limiter = max(min(twoByk * r, 1), 0), where
 * twoByk = 2/k: k = 1 limits most strongly (fully bounded), smaller k relaxes the blend towards
 * central differencing.
 */
void computeLimitedLinearWeights(
    const VolumeField<scalar>& src,
    const SurfaceField<scalar>& flux,
    const SurfaceField<scalar>& cdWeights,
    const SurfaceField<Vec3>& faceDeltaOwner,
    const SurfaceField<Vec3>& faceDeltaNeighbour,
    scalar twoByk,
    SurfaceField<scalar>& weights
);

/* @brief limitedLinear surface interpolation.
 *
 * Registered for scalar fields only. Limiting a vector field is conventionally done on
 * magSqr(phi), which needs a derived scalar field and its gradient; that variant is not
 * implemented here, so a Vec3 field must use another scheme.
 */
template<typename ValueType>
class LimitedLinear :
    public SurfaceInterpolationFactory<ValueType>::template Register<LimitedLinear<ValueType>>
{
    using Base =
        SurfaceInterpolationFactory<ValueType>::template Register<LimitedLinear<ValueType>>;

public:

    LimitedLinear(const Executor& exec, const UnstructuredMesh& mesh, Input input)
        : Base(exec, mesh), geometryScheme_(GeometryScheme::readOrCreate(mesh)),
          twoByk_(readTwoByk(input))
    {
        // Opt in to the per-face cell-to-face offsets while the mesh centres are still alive; the
        // geometry scheme frees them on the first read of any cached geometry field.
        geometryScheme_->ensureFaceDeltas();
    };

    static std::string name() { return "limitedLinear"; }

    static std::string doc() { return "limitedLinear interpolation (TVD limited blend)"; }

    static std::string schema() { return "none"; }

    void interpolate(
        [[maybe_unused]] const VolumeField<ValueType>& src,
        [[maybe_unused]] SurfaceField<ValueType>& dst
    ) const override
    {
        NF_ERROR_EXIT("limitedLinear interpolation scheme requires a faceFlux");
    }

    void interpolate(
        const SurfaceField<scalar>& flux,
        const VolumeField<ValueType>& src,
        SurfaceField<ValueType>& dst
    ) const override
    {
        SurfaceField<scalar> blended(
            src.exec(),
            "limitedLinearWeights",
            src.mesh(),
            createCalculatedBCs<SurfaceBoundary<scalar>>(src.mesh())
        );
        weight(flux, src, blended);
        computeLinearInterpolation(src, blended, dst);
    }

    void weight(const VolumeField<ValueType>&, SurfaceField<scalar>&) const override
    {
        NF_ERROR_EXIT("limitedLinear interpolation scheme requires a faceFlux");
    }

    void weight(
        const SurfaceField<scalar>& flux,
        const VolumeField<ValueType>& src,
        SurfaceField<scalar>& weights
    ) const override
    {
        computeLimitedLinearWeights(
            src,
            flux,
            geometryScheme_->weights(),
            geometryScheme_->faceDeltaOwner(),
            geometryScheme_->faceDeltaNeighbour(),
            twoByk_,
            weights
        );
    }

    std::unique_ptr<SurfaceInterpolationFactory<ValueType>> clone() const override
    {
        return std::make_unique<LimitedLinear>(*this);
    }

private:

    // Scheme spec "limitedLinear <k>": the factory consumed the name, so only k remains. k is
    // written as an integer (e.g. 1) or a fractional value (e.g. 0.2). It is mandatory: a missing
    // or non-numeric coefficient is a spec error and is rejected rather than silently replaced by a
    // default that would change the scheme's behaviour.
    static scalar readTwoByk(Input& input)
    {
        scalar k = 0;
        if (std::holds_alternative<NeoN::TokenList>(input))
        {
            auto& tokens = std::get<NeoN::TokenList>(input);
            if (tokens.peekIs<scalar>())
            {
                k = tokens.next<scalar>();
            }
            else if (tokens.peekIs<label>())
            {
                k = static_cast<scalar>(tokens.next<label>());
            }
            else
            {
                NF_THROW("limitedLinear requires a numeric coefficient: 'limitedLinear <k>'");
            }
        }
        else
        {
            const auto& dict = std::get<NeoN::Dictionary>(input);
            if (!dict.contains("limitedLinearCoeff"))
            {
                NF_THROW("limitedLinear requires the dictionary entry 'limitedLinearCoeff'");
            }
            k = dict.get<scalar>("limitedLinearCoeff");
        }

        if (!(k >= scalar(0) && k <= scalar(1)))
        {
            NF_THROW(
                std::string("limitedLinear coefficient must be in [0, 1], got ") + std::to_string(k)
            );
        }
        return scalar(2) / Kokkos::max(k, ROOTVSMALL);
    }

    const std::shared_ptr<GeometryScheme> geometryScheme_;
    scalar twoByk_;
};

} // namespace NeoN::finiteVolume::cellCentred

namespace NeoN
{

namespace fvcc = finiteVolume::cellCentred;

template class fvcc::LimitedLinear<scalar>;

}
