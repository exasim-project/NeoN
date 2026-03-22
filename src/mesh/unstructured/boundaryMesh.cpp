// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

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
    labelVector isLocal,
    std::vector<localIdx> offset,
    std::vector<localIdx> neighbourRank
)
    : exec_(exec), faceCells_(faceCells), Cf_(cf), Cn_(cn), Sf_(sf), magSf_(magSf), nf_(nf),
      delta_(delta), weights_(weights), deltaCoeffs_(deltaCoeffs), offset_(offset),
      isLocal_(isLocal), neighbourRank_(neighbourRank) {};

// Accessor methods
const labelVector& BoundaryMesh::faceCells() const { return faceCells_; }


template<typename ValueType>
View<const ValueType>
extractSubView(const Vector<ValueType>& vec, const std::vector<localIdx>& offs, localIdx i)
{
    // TODO make offset a Vector<localIdx> instead of std::vector
    auto j = static_cast<std::size_t>(i);
    return vec.view({offs[j], offs[j + 1]});
}


View<const label> BoundaryMesh::faceCells(const localIdx i) const
{
    return extractSubView(faceCells_, offset_, i);
}

const vectorVector& BoundaryMesh::cf() const { return Cf_; }

View<const Vec3> BoundaryMesh::cf(const localIdx i) const
{
    return extractSubView(Cf_, offset_, i);
}

const vectorVector& BoundaryMesh::cn() const { return Cn_; }

View<const Vec3> BoundaryMesh::cn(const localIdx i) const
{
    return extractSubView(Cn_, offset_, i);
}

const vectorVector& BoundaryMesh::sf() const { return Sf_; }

vectorVector& BoundaryMesh::sf() { return Sf_; }

View<const Vec3> BoundaryMesh::sf(const localIdx i) const
{
    return extractSubView(Sf_, offset_, i);
}

const scalarVector& BoundaryMesh::magSf() const { return magSf_; }

View<const scalar> BoundaryMesh::magSf(const localIdx i) const
{
    return extractSubView(magSf_, offset_, i);
}

const vectorVector& BoundaryMesh::nf() const { return nf_; }

vectorVector& BoundaryMesh::nf() { return nf_; }

View<const Vec3> BoundaryMesh::nf(const localIdx i) const
{
    return extractSubView(nf_, offset_, i);
}

const vectorVector& BoundaryMesh::delta() const { return delta_; }

View<const Vec3> BoundaryMesh::delta(const localIdx i) const
{
    return extractSubView(delta_, offset_, i);
}

const scalarVector& BoundaryMesh::weights() const { return weights_; }

View<const scalar> BoundaryMesh::weights(const localIdx i) const
{
    return extractSubView(weights_, offset_, i);
}

const scalarVector& BoundaryMesh::deltaCoeffs() const { return deltaCoeffs_; }

scalar BoundaryMesh::neighbourRank(const localIdx i) const { return neighbourRank_[i]; }

View<const scalar> BoundaryMesh::deltaCoeffs(const localIdx i) const
{
    return extractSubView(deltaCoeffs_, offset_, i);
}

const labelVector& BoundaryMesh::isLocal() const { return isLocal_; }

labelVector& BoundaryMesh::isLocal() { return isLocal_; }

const std::vector<localIdx>& BoundaryMesh::offset() const { return neighbourRank_; }

} // namespace NeoN
