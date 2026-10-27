// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>

#include <Kokkos_Core.hpp>

#include "NeoN/core/error.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/fields/field.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/boundaryContext.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volume/detail/patchSum.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::finiteVolume::cellCentred::volumeBoundary
{

namespace detail
{

// Shared kernel behind FixedMean, mirroring OpenFOAM's fixedMeanFvPatchField::updateCoeffs()
// for a scalar field. The patch value is the adjacent cell value, corrected so that the
// area-weighted mean over the whole patch (all ranks) is the prescribed meanValue:
//   meanPsi = sum(magSf * ownerValue) / sum(magSf)                     (gWeightedAverage)
//   if (|meanValue| > SMALL && |meanPsi| > 0.5 |meanValue|)
//       value = ownerValue * |meanValue| / |meanPsi|                   (scale)
//   else
//       value = ownerValue + (meanValue - meanPsi)                     (shift)
// with SMALL = 1e-15, OpenFOAM's double-precision SMALL. The scaled branch uses the magnitudes
// like OpenFOAM does, so the sign of the owner values is kept. The result is a Dirichlet patch
// (valueFraction = 1, refGrad = 0) recomputed from the current owner values on every call, as
// OpenFOAM does in updateCoeffs().
inline void setFixedMeanValue(
    Field<NeoN::scalar>& domainVector,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range,
    NeoN::scalar meanValue
)
{
    constexpr NeoN::scalar small = 1.0e-15;

    const auto internalV = domainVector.internalVector().view();
    auto [faceAreasV, boundaryFaceOwnersV] =
        views(mesh.boundaryMesh().faceAreas(), mesh.boundaryMesh().faceOwners());

    const NeoN::scalar areaSum = patchSum(
        mesh, range, NEON_LAMBDA(const localIdx i, NeoN::scalar& sum) { sum += faceAreasV[i]; }
    );
    const NeoN::scalar weightedSum = patchSum(
        mesh,
        range,
        NEON_LAMBDA(const localIdx i, NeoN::scalar& sum) {
            sum += faceAreasV[i] * internalV[boundaryFaceOwnersV[i]];
        }
    );

    // gWeightedAverage: no weight, no average (a rank-less or empty patch keeps its owner values)
    const NeoN::scalar meanPsi = areaSum > NeoN::scalar(0) ? weightedSum / areaSum : meanValue;

    const bool scale =
        Kokkos::abs(meanValue) > small && Kokkos::abs(meanPsi) > 0.5 * Kokkos::abs(meanValue);
    const NeoN::scalar factor =
        scale ? Kokkos::abs(meanValue) / Kokkos::abs(meanPsi) : NeoN::scalar(1);
    const NeoN::scalar shift = scale ? NeoN::scalar(0) : meanValue - meanPsi;

    auto [refGradV, valueV, valueFractionV, refValueV] = views(
        domainVector.boundaryData().refGrad(),
        domainVector.boundaryData().value(),
        domainVector.boundaryData().valueFraction(),
        domainVector.boundaryData().refValue()
    );

    NeoN::parallelFor(
        domainVector.exec(),
        range,
        NEON_LAMBDA(const localIdx i) {
            const NeoN::scalar newValue = internalV[boundaryFaceOwnersV[i]] * factor + shift;
            refGradV[i] = NeoN::scalar(0);
            refValueV[i] = newValue;
            valueFractionV[i] = NeoN::scalar(1);
            valueV[i] = newValue;
        },
        "setFixedMeanValue"
    );
}

} // namespace detail

// FixedMean: a Dirichlet patch whose values follow the adjacent cells, shifted or scaled so that
// the area-weighted patch mean is meanValue (OpenFOAM's fixedMean). It is used on a pressure
// outlet where the mean, not the profile, is prescribed. See detail::setFixedMeanValue for the
// exact formula.
//
// The values are refreshed from the adjacent cells on every call, which is OpenFOAM's
// updateCoeffs(). OpenFOAM runs that once, when the matrix of the field is built; the evaluate()
// that follows the solve keeps the values, so the pressure corrector sees the patch values
// computed from the cells *before* the solve. A caller that reproduces this sets the context
// value "evaluateOnly" to a positive number for such an evaluation: the patch keeps its values.
// Without the key the patch always recomputes.
//
// Scalar fields only. meanValue is a plain scalar; OpenFOAM's Function1 form `meanValue constant X`
// has to be flattened by the case reader.
class FixedMean : public VolumeBoundaryFactory<NeoN::scalar>::template Register<FixedMean>
{
    using Base = VolumeBoundaryFactory<NeoN::scalar>::template Register<FixedMean>;

public:

    using Base::correctBoundaryCondition;

    // A Dirichlet patch, like OpenFOAM's (it derives from fixedValue): assignable = false and
    // fixesValue = true. The values are still recomputed from the owner cells on every call.
    FixedMean(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
        : Base(mesh, dict, patchID, {.assignable = false, .fixesValue = true}), mesh_(mesh),
          meanValue_(0)
    {
        if (!dict.contains("meanValue"))
        {
            NF_THROW(
                "fixedMean boundary condition on patch " + std::to_string(patchID)
                + " requires a 'meanValue' entry"
            );
        }
        meanValue_ = dict.get<NeoN::scalar>("meanValue");
    }

    virtual void correctBoundaryCondition(Field<NeoN::scalar>& domainVector) final
    {
        detail::setFixedMeanValue(domainVector, mesh_, this->range(), meanValue_);
    }

    virtual void
    correctBoundaryCondition(Field<NeoN::scalar>& domainVector, const BoundaryContext& ctx) final
    {
        if (ctx.hasValue(evaluateOnlyKey) && ctx.value(evaluateOnlyKey) > NeoN::scalar(0)) return;
        correctBoundaryCondition(domainVector);
    }

    // context value that makes the patch keep the values of its last update
    static constexpr const char* evaluateOnlyKey = "evaluateOnly";

    static std::string name() { return "fixedMean"; }

    std::string getName() const override { return name(); }

    static std::string doc()
    {
        return "Dirichlet patch following the adjacent cell values, shifted or scaled so that the "
               "area-weighted mean over the patch is meanValue (OpenFOAM's fixedMean).";
    }

    static std::string schema() { return "none"; }

    virtual std::unique_ptr<VolumeBoundaryFactory<NeoN::scalar>> clone() const final
    {
        return std::make_unique<FixedMean>(*this);
    }

private:

    const UnstructuredMesh& mesh_;
    NeoN::scalar meanValue_;
};

} // namespace NeoN::finiteVolume::cellCentred::volumeBoundary
