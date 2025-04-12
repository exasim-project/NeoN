// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#include "NeoN/mesh/unstructured/boundaryMesh.hpp"

namespace NeoN
{

BoundaryMesh::BoundaryMesh(
    const Executor& exec,
    labelField faceCells,
    vectorField cf,
    vectorField cn,
    vectorField sf,
    scalarField magSf,
    vectorField nf,
    vectorField delta,
    scalarField weights,
    scalarField deltaCoeffs,
    std::vector<localIdx> offset
)
    : exec_(exec), faceCells_(faceCells), Cf_(cf), Cn_(cn), Sf_(sf), magSf_(magSf), nf_(nf),
      delta_(delta), weights_(weights), deltaCoeffs_(deltaCoeffs), offset_(offset) {};

// Accessor methods
const labelField& BoundaryMesh::faceCells() const { return faceCells_; }


template<typename ValueType>
View<const ValueType>
extractSubSpan(const Field<ValueType>& field, const std::vector<localIdx>& offsets, localIdx i)
{
    auto start = static_cast<size_t>(offsets[i]);
    auto end = static_cast<size_t>(offsets[i + 1]);
    return field.view({start, end});
}


View<const label> BoundaryMesh::faceCells(const localIdx i) const
{
    return extractSubSpan(faceCells_, offset_, i);
}

const vectorField& BoundaryMesh::cf() const { return Cf_; }

View<const Vec3> BoundaryMesh::cf(const localIdx i) const
{
    return extractSubSpan(Cf_, offset_, i);
}

const vectorField& BoundaryMesh::cn() const { return Cn_; }

View<const Vec3> BoundaryMesh::cn(const localIdx i) const
{
    return extractSubSpan(Cn_, offset_, i);
}

const vectorField& BoundaryMesh::sf() const { return Sf_; }

View<const Vec3> BoundaryMesh::sf(const localIdx i) const
{
    return extractSubSpan(Sf_, offset_, i);
}

const scalarField& BoundaryMesh::magSf() const { return magSf_; }

View<const scalar> BoundaryMesh::magSf(const localIdx i) const
{
    return extractSubSpan(magSf_, offset_, i);
}

const vectorField& BoundaryMesh::nf() const { return nf_; }

View<const Vec3> BoundaryMesh::nf(const localIdx i) const
{
    return extractSubSpan(nf_, offset_, i);
}

const vectorField& BoundaryMesh::delta() const { return delta_; }

View<const Vec3> BoundaryMesh::delta(const localIdx i) const
{
    return extractSubSpan(delta_, offset_, i);
}

const scalarField& BoundaryMesh::weights() const { return weights_; }

View<const scalar> BoundaryMesh::weights(const localIdx i) const
{
    return extractSubSpan(weights_, offset_, i);
}

const scalarField& BoundaryMesh::deltaCoeffs() const { return deltaCoeffs_; }

View<const scalar> BoundaryMesh::deltaCoeffs(const localIdx i) const
{
    return extractSubSpan(deltaCoeffs_, offset_, i);
}

const std::vector<localIdx>& BoundaryMesh::offset() const { return offset_; }


} // namespace NeoN
