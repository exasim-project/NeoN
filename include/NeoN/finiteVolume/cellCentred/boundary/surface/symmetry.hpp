// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp>

#include "NeoN/fields/field.hpp"
#include "NeoN/finiteVolume/cellCentred/boundary/surfaceBoundaryFactory.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

namespace NeoN::finiteVolume::cellCentred::surfaceBoundary
{
namespace detail
{
// Scalar specialization: zero flux at the symmetry plane
inline void applySymmetry(
    Field<scalar>& domainVector,
    [[maybe_unused]] const UnstructuredMesh&,
    std::pair<size_t, size_t> range
)
{
    auto [refValueV, valueV] =
        views(domainVector.boundaryData().refValue(), domainVector.boundaryData().value());

    NeoN::parallelFor(
        domainVector.exec(),
        range,
        NEON_LAMBDA(const localIdx i) {
            refValueV[i] = 0;
            valueV[i] = 0;
        },
        "computeSymmetryBoundaryScalar"
    );
}

// Vec3 specialization: zero the normal component of the existing face value
inline void applySymmetry(
    Field<Vec3>& domainVector, const UnstructuredMesh& mesh, std::pair<size_t, size_t> range
)
{
    auto [refValueV, valueV, nHatV] = views(
        domainVector.boundaryData().refValue(),
        domainVector.boundaryData().value(),
        mesh.boundaryMesh().faceUnitNormals()
    );

    NeoN::parallelFor(
        domainVector.exec(),
        range,
        NEON_LAMBDA(const localIdx i) {
            const Vec3 n = nHatV[i];
            const Vec3 v = valueV[i];
            const scalar vn = v & n;
            const Vec3 vtan = v - n * vn;

            refValueV[i] = vtan;
            valueV[i] = vtan;
        },
        "computeSymmetryBoundaryVec3"
    );
}
} // namespace detail


template<typename ValueType>
class Symmetry : public SurfaceBoundaryFactory<ValueType>::template Register<Symmetry<ValueType>>
{
    using Base = typename SurfaceBoundaryFactory<ValueType>::template Register<Symmetry<ValueType>>;

public:

    Symmetry(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
        : Base(mesh, dict, patchID), mesh_(mesh)
    {}

    void correctBoundaryCondition(Field<ValueType>& domainVector) override
    {
        detail::applySymmetry(domainVector, mesh_, this->range());
    }

    static std::string name() { return "symmetry"; }
    static std::string doc()
    {
        return "Symmetry plane (zero gradient for scalars, mirror for vectors)";
    }
    static std::string schema() { return "none"; }

    std::unique_ptr<SurfaceBoundaryFactory<ValueType>> clone() const override
    {
        return std::make_unique<Symmetry>(*this);
    }

private:

    const UnstructuredMesh& mesh_;
};

} // namespace NeoN::finiteVolume::cellCentred::surfaceBoundary
