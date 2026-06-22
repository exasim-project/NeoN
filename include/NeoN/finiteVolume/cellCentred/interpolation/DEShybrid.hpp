// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/fields/field.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/input.hpp"
#include "NeoN/core/tokenList.hpp"
#include "NeoN/core/error.hpp"
#include "NeoN/core/database/fieldCollection.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

#include <memory>
#include <string>

namespace NeoN::finiteVolume::cellCentred
{

/* @brief blends two surface fields face-by-face using a per-face blending factor:
**
**   out_f = (1 - sigma_f) * a_f + sigma_f * b_f
**
** Applied to both the internal and the boundary face data. sigma, a, b and out must share the
** mesh's face layout. Used by DEShybrid for weights (scalar), the interpolated value, and the
** explicit correction (ValueType).
*/
template<typename ValueType>
void blendSurfaceFields(
    const SurfaceField<scalar>& sigma,
    const SurfaceField<ValueType>& a,
    const SurfaceField<ValueType>& b,
    SurfaceField<ValueType>& out
);

/* @class DEShybrid
** @brief Improved hybrid convection scheme of Travin et al. for hybrid RAS/LES calculations
** (OpenFOAM's DEShybrid). It blends two registered sub-schemes by a per-face blending factor
** sigma (0 <= sigma <= sigmaMax):
**
**   weight = (1 - sigma) * w1 + sigma * w2
**
** Scheme 1 should be a low-dissipation scheme (e.g. linear) used in the vortex-resolving LES
** regions; scheme 2 an upwind-biased scheme (e.g. linearUpwind) used in the RAS/irrotational
** regions. Because the weighted face reconstruction w*phi_O + (1-w)*phi_N is linear in w, blending
** the weights is exact for the implicit divergence matrix, and the explicit (deferred) correction
** of a corrected sub-scheme is blended with the same sigma.
**
** This NeoN scheme is deliberately generic and turbulence-agnostic (NeoN has no turbulence-model
** concept): sigma is supplied externally. It is resolved at run time, in priority order, from
**   1. a SurfaceField<scalar> named "<srcName>DEShybridBlendingFactor" registered in src's
**      database (the path used by a solver: NeoFOAM computes sigma from its turbulence model and
**      registers it each step), or
**   2. the member blending factor (set via setBlendingFactor(), default uniform 0 -> pure
**      scheme 1), used for unit tests and standalone explicit interpolation.
**
** The OpenFOAM scheme spec is
**   DEShybrid <scheme1> <scheme2> <delta> <CDES> <U0> <L0> <sigmaMin> <sigmaMax> <OmegaLim>
*[nutLim]
** Only the two sub-scheme names are consumed here; the LES-delta name and the blending-factor
** coefficients drive the sigma computation, which lives on the NeoFOAM side, so they are left for
** that consumer. Sub-scheme trailing tokens (e.g. linearUpwind's grad-scheme name) are cosmetic in
** NeoN and ignored, matching the existing NeoN sub-schemes.
*/
template<typename ValueType>
class DEShybrid :
    public SurfaceInterpolationFactory<ValueType>::template Register<DEShybrid<ValueType>>
{
    using Base = SurfaceInterpolationFactory<ValueType>::template Register<DEShybrid<ValueType>>;

public:

    DEShybrid(const Executor& exec, const UnstructuredMesh& mesh, Input input)
        : Base(exec, mesh), tScheme1_(readSubScheme(exec, mesh, input)),
          tScheme2_(readSubScheme(exec, mesh, input)),
          blendingFactor_(
              exec,
              "DEShybridBlendingFactor",
              mesh,
              createCalculatedBCs<SurfaceBoundary<scalar>>(mesh)
          )
    {
        fill(blendingFactor_.internalVector(), zero<scalar>());
        fill(blendingFactor_.boundaryData().value(), zero<scalar>());
    }

    // Deep-copy: the unique_ptr sub-schemes are cloned so the copy is independent.
    DEShybrid(const DEShybrid& other)
        : Base(other.exec_, other.mesh_), tScheme1_(other.tScheme1_->clone()),
          tScheme2_(other.tScheme2_->clone()), blendingFactor_(other.blendingFactor_)
    {}

    static std::string name() { return "DEShybrid"; }

    static std::string doc() { return "DEShybrid blended interpolation (Travin et al.)"; }

    static std::string schema() { return "none"; }

    void interpolate(
        [[maybe_unused]] const VolumeField<ValueType>& src,
        [[maybe_unused]] SurfaceField<ValueType>& dst
    ) const override
    {
        NF_ERROR_EXIT("DEShybrid interpolation scheme requires a faceFlux");
    }

    void interpolate(
        const SurfaceField<scalar>& flux,
        const VolumeField<ValueType>& src,
        SurfaceField<ValueType>& dst
    ) const override
    {
        const SurfaceField<scalar>& sigma = resolveBlendingFactor(src);
        SurfaceField<ValueType> i1 = makeSurface("DEShybridInterp1");
        SurfaceField<ValueType> i2 = makeSurface("DEShybridInterp2");
        tScheme1_->interpolate(flux, src, i1);
        tScheme2_->interpolate(flux, src, i2);
        blendSurfaceFields(sigma, i1, i2, dst);
    }

    void weight(const VolumeField<ValueType>&, SurfaceField<scalar>&) const override
    {
        NF_ERROR_EXIT("DEShybrid interpolation scheme requires a faceFlux");
    }

    void weight(
        const SurfaceField<scalar>& flux,
        const VolumeField<ValueType>& src,
        SurfaceField<scalar>& weights
    ) const override
    {
        const SurfaceField<scalar>& sigma = resolveBlendingFactor(src);
        SurfaceField<scalar> w1 = makeSurfaceScalar("DEShybridWeight1");
        SurfaceField<scalar> w2 = makeSurfaceScalar("DEShybridWeight2");
        tScheme1_->weight(flux, src, w1);
        tScheme2_->weight(flux, src, w2);
        blendSurfaceFields(sigma, w1, w2, weights);
    }

    // DEShybrid is corrected whenever either sub-scheme carries an explicit correction.
    bool corrected() const override { return tScheme1_->corrected() || tScheme2_->corrected(); }

    void correction(
        const SurfaceField<scalar>& flux,
        const VolumeField<ValueType>& src,
        SurfaceField<ValueType>& corr
    ) const override
    {
        const SurfaceField<scalar>& sigma = resolveBlendingFactor(src);
        // The base default fills zero for uncorrected sub-schemes, so the unconditional calls are
        // safe and yield corr = (1-sigma)*corr1 + sigma*corr2.
        SurfaceField<ValueType> c1 = makeSurface("DEShybridCorr1");
        SurfaceField<ValueType> c2 = makeSurface("DEShybridCorr2");
        tScheme1_->correction(flux, src, c1);
        tScheme2_->correction(flux, src, c2);
        blendSurfaceFields(sigma, c1, c2, corr);
    }

    std::unique_ptr<SurfaceInterpolationFactory<ValueType>> clone() const override
    {
        return std::make_unique<DEShybrid>(*this);
    }

    /* @brief Sets the (fallback) per-face blending factor used when no database-registered field is
    ** found. Copies the internal and boundary data; the supplied field must match the mesh face
    ** layout. Primarily for unit tests and standalone explicit interpolation.
    */
    void setBlendingFactor(const SurfaceField<scalar>& sigma)
    {
        blendingFactor_.internalVector() = sigma.internalVector();
        blendingFactor_.boundaryData() = sigma.boundaryData();
    }

    const SurfaceField<scalar>& blendingFactor() const { return blendingFactor_; }

private:

    // Reads one sub-scheme name from the token stream and constructs it via the runtime-selection
    // factory. Only the scheme name is consumed; any trailing scheme tokens are cosmetic in NeoN
    // (e.g. linearUpwind's grad-scheme name) and remain in the caller's stream, which is harmless
    // because DEShybrid ignores everything after the two sub-scheme names.
    static std::unique_ptr<SurfaceInterpolationFactory<ValueType>>
    readSubScheme(const Executor& exec, const UnstructuredMesh& mesh, Input& input)
    {
        if (!std::holds_alternative<TokenList>(input))
        {
            NF_ERROR_EXIT("DEShybrid requires a TokenList scheme specification");
        }
        std::string schemeName = std::get<TokenList>(input).template next<std::string>();
        TokenList sub;
        sub.insert(schemeName);
        return SurfaceInterpolationFactory<ValueType>::create(exec, mesh, Input(sub));
    }

    // Run-time resolution of sigma: prefer a database-registered field named per the convention,
    // otherwise the member fallback. Any lookup failure falls back to the member field.
    const SurfaceField<scalar>& resolveBlendingFactor(const VolumeField<ValueType>& src) const
    {
        if (src.registered())
        {
            try
            {
                const std::string sigmaName = src.name + "DEShybridBlendingFactor";
                const VectorCollection& col = VectorCollection::instance(src);
                const std::vector<std::string> ids =
                    col.find([&](const Document& d)
                             { return d.get<std::string>("name") == sigmaName; });
                if (!ids.empty())
                {
                    return col.fieldDoc(ids[0]).template field<SurfaceField<scalar>>();
                }
            }
            catch (...)
            {
                // fall through to the member fallback
            }
        }
        return blendingFactor_;
    }

    SurfaceField<ValueType> makeSurface(const std::string& nm) const
    {
        return SurfaceField<ValueType>(
            this->exec_,
            nm,
            this->mesh_,
            createCalculatedBCs<SurfaceBoundary<ValueType>>(this->mesh_)
        );
    }

    SurfaceField<scalar> makeSurfaceScalar(const std::string& nm) const
    {
        return SurfaceField<scalar>(
            this->exec_, nm, this->mesh_, createCalculatedBCs<SurfaceBoundary<scalar>>(this->mesh_)
        );
    }

    std::unique_ptr<SurfaceInterpolationFactory<ValueType>> tScheme1_;
    std::unique_ptr<SurfaceInterpolationFactory<ValueType>> tScheme2_;
    SurfaceField<scalar> blendingFactor_;
};

} // namespace NeoN::finiteVolume::cellCentred

namespace NeoN
{

namespace fvcc = finiteVolume::cellCentred;

template class fvcc::DEShybrid<scalar>;
template class fvcc::DEShybrid<Vec3>;

}
