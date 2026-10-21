// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/error.hpp"
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
    SurfaceField<scalar>& weights,
    bool cellLimitedGradient = false
);

/* @brief limitedLinear surface interpolation.
 *
 * Registered for scalar fields only. Limiting a vector field is conventionally done on
 * magSqr(phi), which needs a derived scalar field and its gradient; that variant is not
 * implemented here, so a Vec3 field must use another scheme.
 *
 * Spec: "limitedLinear <k> [cellLimited]". The optional trailing word selects the gradient the
 * TVD ratio is built from. That gradient is a property of the transported *field*, not of this
 * scheme -- a framework that keeps a per-field table of gradient schemes resolves it there and
 * states the outcome here, because this class has no field-to-scheme table to consult. Omitting
 * it keeps the unlimited Gauss-Green gradient, which reports a larger upwind-cell slope, hence a
 * larger TVD ratio and a limiter closer to pure central differencing.
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
          twoByk_(readTwoByk(input)), cellLimitedGradient_(readCellLimited(input))
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
            weights,
            cellLimitedGradient_
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

    // Optional trailing marker "limitedLinear <k> cellLimited", which selects the cell-limited
    // (minmod, k=1) gradient for the TVD ratio; absent, the unlimited Gauss-Green gradient is
    // used. An unrecognised trailing word is a spec error rather than a silently ignored token.
    static bool readCellLimited(Input& input)
    {
        if (std::holds_alternative<NeoN::TokenList>(input))
        {
            auto& tokens = std::get<NeoN::TokenList>(input);
            if (!tokens.peekIs<std::string>()) return false;

            const auto word = tokens.next<std::string>();
            if (word == "cellLimited") return true;
            NF_THROW(
                std::string("limitedLinear accepts only 'cellLimited' after its coefficient, got ")
                + word
            );
        }

        const auto& dict = std::get<NeoN::Dictionary>(input);
        if (!dict.contains("limitedLinearGrad")) return false;

        const auto word = dict.get<std::string>("limitedLinearGrad");
        if (word == "cellLimited") return true;
        if (word == "Gauss") return false;
        NF_THROW(std::string("limitedLinearGrad must be 'Gauss' or 'cellLimited', got ") + word);
    }

    const std::shared_ptr<GeometryScheme> geometryScheme_;
    scalar twoByk_;
    bool cellLimitedGradient_;
};

} // namespace NeoN::finiteVolume::cellCentred

namespace NeoN
{

namespace fvcc = finiteVolume::cellCentred;

template class fvcc::LimitedLinear<scalar>;

}
