// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/database/fieldCollection.hpp"
#include "NeoN/fields/field.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
// #include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"

namespace NeoN::finiteVolume::cellCentred::volumeBoundary
{
namespace fvcc = NeoN::finiteVolume::cellCentred;
namespace detail
{

static constexpr label maxIter = 10;
static constexpr scalar tolerance = 1e-9;
static constexpr scalar kappa = 0.41;
static constexpr scalar E = 9.8;

KOKKOS_INLINE_FUNCTION
scalar computeUTau(
    const scalar magGradU,
    const scalar magUp,
    const scalar y,
    const scalar nuw,
    const scalar nutw,
    scalar& err
)
{
    err = 0.0;

    scalar ut = Kokkos::sqrt((nutw + nuw) * magGradU);
    if (ut <= ROOTVSMALL)
    {
        return 0.0;
    }

    int iter = 0;
    do
    {
        const scalar kUu = Kokkos::min(kappa * magUp / ut, scalar(50));
        const scalar fkUu = Kokkos::exp(kUu) - 1.0 - kUu * (1.0 + 0.5 * kUu);

        const scalar f =
            -ut * y / nuw + magUp / ut + (1.0 / E) * (fkUu - (1.0 / 6.0) * kUu * kUu * kUu);

        const scalar df = y / nuw + magUp / (ut * ut) + (1.0 / E) * kUu * fkUu / ut;

        const scalar uTauNew = ut + f / df;
        err = NeoN::mag((ut - uTauNew) / ut);
        ut = uTauNew;
    }
    while (ut > ROOTVSMALL && err > tolerance && ++iter < maxIter);

    return ut > 0.0 ? ut : 0.0;
}
/*
template<typename ValueType>
const VolumeField<ValueType>& lookupVolumeField(
    const Field<scalar>& field,
    const std::string& fieldName
)
{
    if (!field.hasDatabase())
    {
        throw std::runtime_error {
            "Database not set: make sure the field is registered in the database"
        };
    }
    if (field.fieldCollectionName().empty())
    {
        throw std::runtime_error {
            "Field collection name not set: make sure the field is registered in the database"
        };
    }
    const auto& collection = VectorCollection::instance(field.db(), field.fieldCollectionName());
    const auto matches = collection.find(
        [&fieldName](const NeoN::Document& doc) { return NeoN::name(doc) == fieldName; }
    );

    if (matches.empty())
    {
        throw std::runtime_error {"VectorCollection does not contain field '" + fieldName + "'"};
    }

    return collection.fieldDoc(matches.front()).field<VolumeField<ValueType>>();
}
*/
inline void setNutUSpaldingWallFunction(
    Field<scalar>& domainVector,
    const fvcc::VolumeField<Vec3>& U,
    const fvcc::VolumeField<scalar>& nu,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range
)
{
    const auto uInternal = U.internalVector().view();
    const auto uBoundary = U.boundaryData().value().view();
    const auto nuBoundary = nu.boundaryData().value().view();

    auto [refGradient, value, valueFraction, refValue, faceCells, deltaCoeffs, delta] = views(
        domainVector.boundaryData().refGrad(),
        domainVector.boundaryData().value(),
        domainVector.boundaryData().valueFraction(),
        domainVector.boundaryData().refValue(),
        mesh.boundaryMesh().faceCells(),
        mesh.boundaryMesh().deltaCoeffs(),
        mesh.boundaryMesh().delta()
    );

    NeoN::parallelFor(
        domainVector.exec(),
        range,
        NEON_LAMBDA(const localIdx i) {
            const localIdx owner = faceCells[i];

            const Vec3 uInt = uInternal[owner];
            const Vec3 uWall = uBoundary[i];
            const Vec3 diff = uWall - uInt;

            const scalar magUp = NeoN::mag(diff);
            const scalar magGradU = NeoN::mag(diff * deltaCoeffs[i]);
            const scalar y = NeoN::mag(delta[i]);
            const scalar nuw = nuBoundary[i];

            const scalar currentNut = value[i];

            scalar err = 0.0;
            const scalar uTau = computeUTau(magGradU, magUp, y, nuw, currentNut, err);

            const scalar nutCandidate = (uTau * uTau) / (magGradU + ROOTVSMALL) - nuw;

            const scalar nutw = nutCandidate > 0.0 ? nutCandidate : 0.0;

            refValue[i] = nutw;
            value[i] = nutw;
            valueFraction[i] = 1.0;
            refGradient[i] = 0.0;
        },
        "setNutUSpaldingWallFunction"
    );
}

} // namespace detail

class NutUSpaldingWallFunction :
    public VolumeBoundaryFactory<scalar>::template Register<NutUSpaldingWallFunction>
{
    using Base = VolumeBoundaryFactory<scalar>::template Register<NutUSpaldingWallFunction>;

public:

    NutUSpaldingWallFunction(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
        : Base(mesh, dict, patchID, {.assignable = false, .fixesValue = true}), mesh_(mesh)
    {}

    void correctBoundaryCondition(Field<scalar>& domainVector) final
    {
        // const auto& uField = detail::lookupVolumeField<Vec3>(domainVector, "U");
        // const auto& nuField = detail::lookupVolumeField<scalar>(domainVector, "nu");
        auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::scalar>>(mesh_);
        auto volumeVec3BCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<NeoN::Vec3>>(mesh_);
        fvcc::VolumeField<NeoN::scalar> nuField(domainVector.exec(), "nu", mesh_, volumeBCs);
        fvcc::VolumeField<NeoN::Vec3> uField(domainVector.exec(), "uField", mesh_, volumeVec3BCs);
        detail::setNutUSpaldingWallFunction(domainVector, uField, nuField, mesh_, this->range());
    }

    static std::string name() { return "nutUSpaldingWallFunction"; }

    static std::string doc()
    {
        return "Spalding wall-function for nut with fixed internal constants.";
    }

    static std::string schema() { return "none"; }

    std::unique_ptr<VolumeBoundaryFactory<scalar>> clone() const final
    {
        return std::make_unique<NutUSpaldingWallFunction>(*this);
    }

private:

    const UnstructuredMesh& mesh_;
};

} // namespace NeoN::finiteVolume::cellCentred::volumeBoundary
