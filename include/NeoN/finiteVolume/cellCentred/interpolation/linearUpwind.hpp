// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/fields/field.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/primitives/tensor.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/surfaceInterpolation.hpp"
#include "NeoN/finiteVolume/cellCentred/interpolation/upwind.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/finiteVolume/cellCentred/stencil/geometryScheme.hpp"

#include <Kokkos_Core.hpp>

#include <functional>
#include <string>

namespace NeoN::finiteVolume::cellCentred
{

namespace detail
{

/* @brief maps a face-interpolated field type to the type of its cell gradient:
** a scalar field has a Vec3 gradient, a Vec3 field has a Tensor gradient.
*/
template<typename ValueType>
struct LinearUpwindGradType;

template<>
struct LinearUpwindGradType<scalar>
{
    using type = Vec3;
};

template<>
struct LinearUpwindGradType<Vec3>
{
    using type = Tensor;
};

} // namespace detail

/* @brief computational kernel to perform a linearUpwind interpolation
** from a source volume field to a surface field. The face value is the upwind
** cell value plus a gradient-based correction:
**
**   phi_f = phi_U + (Cf - C_U) & grad(phi)_U
**
** where U is the upwind cell (owner if faceFlux >= 0, neighbour otherwise),
** Cf is the face centre and C_U the upwind cell centre. The gradient is computed
** internally with a Gauss-Green scheme. This reduces to plain upwind where the
** field is locally constant and is second-order accurate for smooth fields.
**
** @pre src.correctBoundaryConditions() must have been called by the caller so that
**      processor-boundary ghost values are up to date before the gradient is computed.
** @param src the input field
** @param flux the face flux determining the upwind direction
** @param faceDeltaOwner Cf - C_owner per internal face (from the geometry scheme)
** @param faceDeltaNeighbour Cf - C_neighbour per internal face (from the geometry scheme)
** @param weights linear-interpolation owner-side weight per internal face (from the geometry
**        scheme); only read when ofStyleVectorClip is set.
** @param dst the target surface field
** @param ofStyleVectorClip when true (the "linearUpwindV" scheme, Vec3 fields only), the
**        unlimited correction at each face is additionally clipped to match OpenFOAM's
**        linearUpwindV::correction() exactly: bound the correction vector, as a whole (all
**        components scaled together, not independently), by that face's own upwind/downwind
**        value jump -- see applyLinearUpwind for the derivation. Ignored for scalar fields.
*/
template<typename ValueType>
void computeLinearUpwindInterpolation(
    const VolumeField<ValueType>& src,
    const SurfaceField<scalar>& flux,
    const SurfaceField<Vec3>& faceDeltaOwner,
    const SurfaceField<Vec3>& faceDeltaNeighbour,
    const SurfaceField<scalar>& weights,
    SurfaceField<ValueType>& dst,
    const bool ofStyleVectorClip = false
);

/* @brief computes only the gradient correction part of the linearUpwind face value,
** `(Cf - C_U) & grad(phi)_U`, into dst (without the upwind cell value). Boundary faces are set to
** zero, matching OpenFOAM which corrects only coupled patches. Used for the implicit deferred
** correction (the explicit RHS source added by the divergence operator).
**
** @pre src.correctBoundaryConditions() must have been called by the caller so that
**      processor-boundary ghost values are up to date before the gradient is computed.
*/
template<typename ValueType>
void computeLinearUpwindCorrection(
    const VolumeField<ValueType>& src,
    const SurfaceField<scalar>& flux,
    const SurfaceField<Vec3>& faceDeltaOwner,
    const SurfaceField<Vec3>& faceDeltaNeighbour,
    const SurfaceField<scalar>& weights,
    SurfaceField<ValueType>& dst,
    const bool ofStyleVectorClip = false
);

// @tparam CellLimited selects the "linearUpwindV" scheme (Vec3 fields only): the unlimited
// Gauss-Green gradient correction at each face is clipped to match OpenFOAM's own
// linearUpwindV::correction() exactly: a per-face bound on the correction vector as a whole.
// OpenFOAM does not pre-limit the gradient the way a generic cellLimited/minmod gradient scheme
// would; it limits the resulting face correction, face by face, against that face's own
// immediate upwind/downwind value jump. The name CellLimited is historical (an earlier version
// built a cell-wide minmod-limited gradient, a different algorithm) and is kept to avoid an
// unrelated rename across call sites.
template<typename ValueType, bool CellLimited = false>
class LinearUpwind :
    public SurfaceInterpolationFactory<ValueType>::template Register<
        LinearUpwind<ValueType, CellLimited>>
{
    using Base = SurfaceInterpolationFactory<ValueType>::template Register<
        LinearUpwind<ValueType, CellLimited>>;

public:

    LinearUpwind(const Executor& exec, const UnstructuredMesh& mesh, Input input)
        : Base(exec, mesh), geometryScheme_(GeometryScheme::readOrCreate(mesh)),
          gradSchemeName_(readGradSchemeName(input)),
          cellLimitedGradient_(readCellLimited(gradSchemeName_))
    {
        // Opt in to the geometry scheme's per-internal-face cell-to-face offset vectors. These are
        // computed lazily and only for schemes that need them, so non-linearUpwind runs never
        // allocate them. Done in the constructor — while the mesh centres are still alive — because
        // the geometry scheme frees those centres on the first read of any cached geometry field.
        geometryScheme_->ensureFaceDeltas();
    };

    static std::string name() { return CellLimited ? "linearUpwindV" : "linearUpwind"; }

    static std::string doc()
    {
        return CellLimited ? "linearUpwindV interpolation (per-face bounded correction, "
                             "matching OpenFOAM's linearUpwindV)"
                           : "linearUpwind interpolation";
    }

    static std::string schema() { return "none"; }

    void interpolate(
        [[maybe_unused]] const VolumeField<ValueType>& src,
        [[maybe_unused]] SurfaceField<ValueType>& dst
    ) const override
    {
        NF_ERROR_EXIT("linearUpwind interpolation scheme requires a faceFlux");
    }

    void interpolate(
        const SurfaceField<scalar>& flux,
        const VolumeField<ValueType>& src,
        SurfaceField<ValueType>& dst
    ) const override
    {
        computeLinearUpwindInterpolation(
            src,
            flux,
            geometryScheme_->faceDeltaOwner(),
            geometryScheme_->faceDeltaNeighbour(),
            geometryScheme_->weights(),
            dst,
            cellLimitedGradient_
        );
    }

    // linearUpwind shares upwind's implicit weights; the gradient correction is an explicit
    // (deferred) term applied through interpolate() above, mirroring OpenFOAM's corrected scheme.
    void weight(const VolumeField<ValueType>&, SurfaceField<scalar>&) const override
    {
        NF_ERROR_EXIT("linearUpwind interpolation scheme requires a faceFlux");
    }

    void weight(
        const SurfaceField<scalar>& faceFlux,
        const VolumeField<ValueType>& src,
        SurfaceField<scalar>& weights
    ) const override
    {
        computeUpwindInterpolationWeights(faceFlux, src, weights);
    }

    // linearUpwind carries an explicit gradient correction beyond the upwind weights.
    bool corrected() const override { return true; }

    void correction(
        const SurfaceField<scalar>& faceFlux,
        const VolumeField<ValueType>& src,
        SurfaceField<ValueType>& corr
    ) const override
    {
        computeLinearUpwindCorrection(
            src,
            faceFlux,
            geometryScheme_->faceDeltaOwner(),
            geometryScheme_->faceDeltaNeighbour(),
            geometryScheme_->weights(),
            corr,
            cellLimitedGradient_
        );
    }

    std::unique_ptr<SurfaceInterpolationFactory<ValueType>> clone() const override
    {
        return std::make_unique<LinearUpwind>(*this);
    }

private:

    // The scheme spec is "linearUpwind <gradScheme>", e.g.
    // "linearUpwind cellLimited Gauss linear 1". The deferred correction has to be built from
    // the gradient named there: falling back to the unlimited Gauss-Green gradient leaves the
    // reconstruction unclipped, a systematically more aggressive convection term. So the
    // leading word decides whether the correction is cell-limited.
    //
    // A bare key ("linearUpwind limited", naming an entry in the caller's gradient-scheme
    // table) cannot be resolved here -- NeoN holds no such table -- and falls back to the
    // unlimited gradient. Callers that own the scheme dictionary should expand such a key to
    // its definition before handing the spec over.
    static std::string readGradSchemeName(Input& input)
    {
        if (std::holds_alternative<NeoN::TokenList>(input))
        {
            auto& tokens = std::get<NeoN::TokenList>(input);
            if (tokens.peekIs<std::string>())
            {
                return tokens.next<std::string>();
            }
        }
        return "Gauss";
    }

    // linearUpwindV forces it; otherwise honour the gradient named in the spec.
    static bool readCellLimited(const std::string& gradSchemeName)
    {
        return CellLimited || gradSchemeName == "cellLimited";
    }

    const std::shared_ptr<GeometryScheme> geometryScheme_;
    std::string gradSchemeName_;
    bool cellLimitedGradient_;
};

} // namespace NeoN::finiteVolume::cellCentred

namespace NeoN
{

namespace fvcc = finiteVolume::cellCentred;

template class fvcc::LinearUpwind<scalar>;
template class fvcc::LinearUpwind<Vec3>;
// linearUpwindV: cell-limited gradient reconstruction, vector fields only.
template class fvcc::LinearUpwind<Vec3, true>;

}
