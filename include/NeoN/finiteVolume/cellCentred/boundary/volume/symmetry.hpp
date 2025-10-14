// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
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
void setSymmetryValue(
    Field<ValueType>& domainVector,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range
);

// --- Scalar specialization ---
template<>
inline void setSymmetryValue<NeoN::scalar>(
    Field<NeoN::scalar>& domainVector,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range
)
{
    const auto internal = domainVector.internalVector().view();

    auto [refGrad, value, valueFraction, refValue, faceCells] = views(
        domainVector.boundaryData().refGrad(),
        domainVector.boundaryData().value(),
        domainVector.boundaryData().valueFraction(),
        domainVector.boundaryData().refValue(),
        mesh.boundaryMesh().faceCells()
    );

    NeoN::parallelFor(
        domainVector.exec(),
        range,
        KOKKOS_LAMBDA(const localIdx i) {
            const localIdx owner = faceCells[i];
            const auto v = internal[owner];

            // Scalar symmetry → zero-gradient
            refValue[i] = v;
            value[i] = v;
            valueFraction[i] = 0.0;
            refGrad[i] = 0.0;
        },
        "setSymmetryValue(scalar)"
    );
}

// --- Vec3 specialization ---
template<>
inline void setSymmetryValue<NeoN::Vec3>(
    Field<NeoN::Vec3>& domainVector,
    const UnstructuredMesh& mesh,
    std::pair<localIdx, localIdx> range
)
{
    const auto internal = domainVector.internalVector().view();

    auto [refGrad, value, valueFraction, refValue, faceCells, nHat] = views(
        domainVector.boundaryData().refGrad(),
        domainVector.boundaryData().value(),
        domainVector.boundaryData().valueFraction(),
        domainVector.boundaryData().refValue(),
        mesh.boundaryMesh().faceCells(),
        mesh.boundaryMesh().nf()
    );

    NeoN::parallelFor(
        domainVector.exec(),
        range,
        KOKKOS_LAMBDA(const localIdx i) {
            const localIdx owner = faceCells[i];
            const auto v = internal[owner];
            const auto n = nHat[i];

            // Tangential projection (remove normal component)
            const auto vn = v & n;
            const auto vtan = v - n * vn;

            refValue[i] = vtan;
            value[i] = vtan;
            valueFraction[i] = 0.0;
            refGrad[i] = NeoN::zero<NeoN::Vec3>();
        },
        "setSymmetryValue(Vec3)"
    );
}

} // namespace detail


template<typename ValueType>
class Symmetry : public VolumeBoundaryFactory<ValueType>::template Register<Symmetry<ValueType>>
{
    using Base = typename VolumeBoundaryFactory<ValueType>::template Register<Symmetry<ValueType>>;

public:

    using SymmetryType = Symmetry<ValueType>;

    Symmetry(const UnstructuredMesh& mesh, const Dictionary& dict, localIdx patchID)
        : Base(mesh, dict, patchID, {.assignable = false}), mesh_(mesh)
    {}

    virtual void correctBoundaryCondition(Field<ValueType>& domainVector) final
    {
        detail::setSymmetryValue(domainVector, mesh_, this->range());
    }

    static std::string name() { return "symmetry"; }

    static std::string doc()
    {
        return "Symmetry plane (scalar: zero-gradient; vector: tangential projection).";
    }

    static std::string schema() { return "none"; }

    virtual std::unique_ptr<VolumeBoundaryFactory<ValueType>> clone() const final
    {
        return std::make_unique<Symmetry>(*this);
    }

private:

    const UnstructuredMesh& mesh_;
};

} // namespace NeoN::finiteVolume::cellCentred::volumeBoundary
