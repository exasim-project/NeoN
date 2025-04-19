// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#include "NeoN/mesh/unstructured/boundaryMesh.hpp"

namespace NeoN
{

BoundaryMesh::BoundaryMesh(
    const Executor& exec,
    labelVector faceCells,
    vectorVector cf,
    vectorVector cn,
    vectorVector sf,
    scalarVector magSf,
    vectorVector nf,
    vectorVector delta,
    scalarVector weights,
    scalarVector deltaCoeffs,
    std::vector<localIdx> offset
)
    : exec_(exec), faceCells_(faceCells), Cf_(cf), Cn_(cn), Sf_(sf), magSf_(magSf), nf_(nf),
      delta_(delta), weights_(weights), deltaCoeffs_(deltaCoeffs), offset_(offset) {};

// Accessor methods
const labelVector& BoundaryMesh::faceCells() const { return faceCells_; }


template<typename ValueType>
View<const ValueType>
extractSubSpan(const Vector<ValueType>& vec, const std::vector<localIdx>& offs, localIdx i)
{
    // FIXME make offset a Vector<localIdx> instead of std::vector
    auto j = static_cast<std::size_t>(i);
    return vec.view({offs[j], offs[j + 1]});
}


View<const label> BoundaryMesh::faceCells(const localIdx i) const
{
    return extractSubSpan(faceCells_, offset_, i);
}

const vectorVector& BoundaryMesh::cf() const { return Cf_; }

View<const Vec3> BoundaryMesh::cf(const localIdx i) const
{
    return extractSubSpan(Cf_, offset_, i);
}

const vectorVector& BoundaryMesh::cn() const { return Cn_; }

View<const Vec3> BoundaryMesh::cn(const localIdx i) const
{
    return extractSubSpan(Cn_, offset_, i);
}

const vectorVector& BoundaryMesh::sf() const { return Sf_; }

View<const Vec3> BoundaryMesh::sf(const localIdx i) const
{
    return extractSubSpan(Sf_, offset_, i);
}

const scalarVector& BoundaryMesh::magSf() const { return magSf_; }

View<const scalar> BoundaryMesh::magSf(const localIdx i) const
{
    return extractSubSpan(magSf_, offset_, i);
}

const vectorVector& BoundaryMesh::nf() const { return nf_; }

View<const Vec3> BoundaryMesh::nf(const localIdx i) const
{
    return extractSubSpan(nf_, offset_, i);
}

const vectorVector& BoundaryMesh::delta() const { return delta_; }

View<const Vec3> BoundaryMesh::delta(const localIdx i) const
{
    return extractSubSpan(delta_, offset_, i);
}

const scalarVector& BoundaryMesh::weights() const { return weights_; }

View<const scalar> BoundaryMesh::weights(const localIdx i) const
{
    return extractSubSpan(weights_, offset_, i);
}

const scalarVector& BoundaryMesh::deltaCoeffs() const { return deltaCoeffs_; }

View<const scalar> BoundaryMesh::deltaCoeffs(const localIdx i) const
{
    return extractSubSpan(deltaCoeffs_, offset_, i);
}

const std::vector<localIdx>& BoundaryMesh::offset() const { return offset_; }


} // namespace NeoN
