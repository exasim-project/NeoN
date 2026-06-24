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
** @param src the input field
** @param flux the face flux determining the upwind direction
** @param faceDeltaOwner Cf - C_owner per internal face (from the geometry scheme)
** @param faceDeltaNeighbour Cf - C_neighbour per internal face (from the geometry scheme)
** @param dst the target surface field
*/
template<typename ValueType>
void computeLinearUpwindInterpolation(
    const VolumeField<ValueType>& src,
    const SurfaceField<scalar>& flux,
    const SurfaceField<Vec3>& faceDeltaOwner,
    const SurfaceField<Vec3>& faceDeltaNeighbour,
    SurfaceField<ValueType>& dst,
    const bool cellLimitedGradient = false
);

/* @brief computes only the gradient correction part of the linearUpwind face value,
** `(Cf - C_U) & grad(phi)_U`, into dst (without the upwind cell value). Boundary faces are set to
** zero, matching OpenFOAM which corrects only coupled patches. Used for the implicit deferred
** correction (the explicit RHS source added by the divergence operator).
*/
template<typename ValueType>
void computeLinearUpwindCorrection(
    const VolumeField<ValueType>& src,
    const SurfaceField<scalar>& flux,
    const SurfaceField<Vec3>& faceDeltaOwner,
    const SurfaceField<Vec3>& faceDeltaNeighbour,
    SurfaceField<ValueType>& dst,
    const bool cellLimitedGradient = false
);

// @tparam CellLimited when true the gradient correction is built from the cell-limited (minmod,
// k=1) gradient rather than the unlimited Gauss-Green gradient — this is the "linearUpwindV"
// scheme, matching OpenFOAM's directionally-bounded vector reconstruction. Only meaningful for
// Vec3 fields; the scalar specialisation ignores it.
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
          gradSchemeName_(readGradSchemeName(input))
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
        return CellLimited ? "linearUpwindV interpolation (cell-limited gradient correction)"
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
            dst,
            CellLimited
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
            corr,
            CellLimited
        );
    }

    InlineWeightKernel inlineWeightKernel(const SurfaceField<scalar>& /*flux*/) const override
    {
        return UpwindInlineKernel {};
    }

    std::unique_ptr<SurfaceInterpolationFactory<ValueType>> clone() const override
    {
        return std::make_unique<LinearUpwind>(*this);
    }

private:

    // The OpenFOAM scheme spec is "linearUpwind <gradScheme>"; NeoN currently has a single
    // (Gauss-Green) gradient scheme, so the name is read for spec-compatibility but not used to
    // select a scheme. Returns a default when absent.
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

    const std::shared_ptr<GeometryScheme> geometryScheme_;
    std::string gradSchemeName_;
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
