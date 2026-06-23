// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp>

#include "NeoN/fields/field.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/volumeBoundaryFactory.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

namespace NeoN::finiteVolume::cellCentred::volumeBoundary
{

namespace detail
{

// Primary declaration
template<typename ValueType>
void setSlipValue(
    Field<ValueType>& domainVector,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range
);

// --- Scalar specialization: zero-gradient ---
template<>
inline void setSlipValue<NeoN::scalar>(
    Field<NeoN::scalar>& domainVector,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range
)
{
    const auto internalV = domainVector.internalVector().view();

    auto [refGradV, valueV, valueFractionV, refValueV, boundaryFaceOwnersV] = views(
        domainVector.boundaryData().refGrad(),
        domainVector.boundaryData().value(),
        domainVector.boundaryData().valueFraction(),
        domainVector.boundaryData().refValue(),
        mesh.boundaryMesh().faceOwners()
    );

    NeoN::parallelFor(
        domainVector.exec(),
        range,
        NEON_LAMBDA(const localIdx i) {
            const localIdx owner = boundaryFaceOwnersV[i];
            const auto v = internalV[owner];

            refValueV[i] = v;
            valueV[i] = v;
            valueFractionV[i] = 0.0;
            refGradV[i] = 0.0;
        },
        "setSlipValue(scalar)"
    );
}

// --- Vec3 specialization: tangential projection (remove normal component) ---
template<>
inline void setSlipValue<NeoN::Vec3>(
    Field<NeoN::Vec3>& domainVector,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range
)
{
    const auto internalV = domainVector.internalVector().view();

    auto [refGradV, valueV, valueFractionV, refValueV, boundaryFaceOwnersV, faceUnitNormalsV] =
        views(
            domainVector.boundaryData().refGrad(),
            domainVector.boundaryData().value(),
            domainVector.boundaryData().valueFraction(),
            domainVector.boundaryData().refValue(),
            mesh.boundaryMesh().faceOwners(),
            mesh.boundaryMesh().faceUnitNormals()
        );

    NeoN::parallelFor(
        domainVector.exec(),
        range,
        NEON_LAMBDA(const localIdx i) {
            const localIdx owner = boundaryFaceOwnersV[i];
            const auto v = internalV[owner];
            const auto n = faceUnitNormalsV[i];

            const auto vtan = v - n * (v & n);

            refValueV[i] = vtan;
            valueV[i] = vtan;
            valueFractionV[i] = 0.0;
            refGradV[i] = NeoN::zero<NeoN::Vec3>();
        },
        "setSlipValue(Vec3)"
    );
}

} // namespace detail


template<typename ValueType>
class Slip : public VolumeBoundaryFactory<ValueType>::template Register<Slip<ValueType>>
{
    using Base = typename VolumeBoundaryFactory<ValueType>::template Register<Slip<ValueType>>;

public:

    using Base::correctBoundaryCondition;

    using SlipType = Slip<ValueType>;

    Slip(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
        : Base(mesh, dict, patchID, {.assignable = false, .fixesValue = false}), mesh_(mesh)
    {}

    virtual void correctBoundaryCondition(Field<ValueType>& domainVector) final
    {
        detail::setSlipValue(domainVector, mesh_, this->range());
    }

    static std::string name() { return "slip"; }

    std::string getName() const override { return name(); }

    static std::string doc()
    {
        return "Slip wall (scalar: zero-gradient; vector: tangential projection).";
    }

    static std::string schema() { return "none"; }

    virtual std::unique_ptr<VolumeBoundaryFactory<ValueType>> clone() const final
    {
        return std::make_unique<Slip>(*this);
    }

private:

    const UnstructuredMesh& mesh_;
};

} // namespace NeoN::finiteVolume::cellCentred::volumeBoundary
