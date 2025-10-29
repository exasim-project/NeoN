// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
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
// Scalar specialization --------------------------------------------------------
inline void applySymmetry(
    Field<scalar>& domainVector, const UnstructuredMesh& mesh, std::pair<size_t, size_t> range
)
{
    auto [refValueV, valueV, internalValuesV, faceCellsV] = views(
        domainVector.boundaryData().refValue(),
        domainVector.boundaryData().value(),
        domainVector.internalVector(),
        mesh.boundaryMesh().faceCells()
    );

    const localIdx nInternalFaces = mesh.nInternalFaces();

    NeoN::parallelFor(
        domainVector.exec(),
        range,
        KOKKOS_LAMBDA(const localIdx i) {
            const localIdx owner = faceCellsV[i];
            const scalar v = internalValuesV[owner];

            refValueV[i] = v;
            valueV[i] = v;
        }
    );
}

// Vec3 specialization ----------------------------------------------------------
inline void applySymmetry(
    Field<Vec3>& domainVector, const UnstructuredMesh& mesh, std::pair<size_t, size_t> range
)
{
    auto [refValueV, valueV, internalValuesV, faceCellsV, nHatV] = views(
        domainVector.boundaryData().refValue(),
        domainVector.boundaryData().value(),
        domainVector.internalVector(),
        mesh.boundaryMesh().faceCells(),
        mesh.boundaryMesh().nf()
    );

    const localIdx nInternalFaces = mesh.nInternalFaces();

    NeoN::parallelFor(
        domainVector.exec(),
        range,
        KOKKOS_LAMBDA(const localIdx i) {
            const localIdx owner = faceCellsV[i];
            const Vec3 n = nHatV[i];

            Vec3 v = internalValuesV[owner];
            const scalar vn = v & n; // dot product
            const auto vtan = v - n * vn;

            refValueV[i] = vtan;
            valueV[i] = vtan;
        }
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
