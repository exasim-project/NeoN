// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/boundaryMesh.hpp"


namespace NeoN
{

void BoundaryMesh::validate() const {}

BoundaryMesh::BoundaryMesh(
    const Executor& exec,
    labelVector faceOwners,
    vectorVector faceCenters,
    vectorVector ownerCellCenters,
    vectorVector faceNormals,
    scalarVector faceAreas,
    vectorVector faceUnitNormals,
    vectorVector delta,
    scalarVector weights,
    scalarVector deltaCoeffs,
    std::vector<localIdx> offset,
    localIdx procBoundaryPatches,
    std::vector<localIdx> neighbourRank
)
    : exec_(exec), faceOwners_(faceOwners), faceCenters_(faceCenters),
      ownerCellCenters_(ownerCellCenters), faceNormals_(faceNormals), faceAreas_(faceAreas),
      faceUnitNormals_(faceUnitNormals), delta_(delta), weights_(weights),
      deltaCoeffs_(deltaCoeffs), offset_(offset), procBoundaryPatches_(procBoundaryPatches),
      neighbourRank_(neighbourRank) {};

// Accessor methods
const labelVector& BoundaryMesh::faceOwners() const { return faceOwners_; }


template<typename ValueType>
View<const ValueType>
extractSubView(const Vector<ValueType>& vec, const std::vector<localIdx>& offs, localIdx i)
{
    // TODO make offset a Vector<localIdx> instead of std::vector
    auto j = static_cast<std::size_t>(i);
    return vec.view({offs[j], offs[j + 1]});
}


View<const label> BoundaryMesh::faceOwners(const localIdx i) const
{
    return extractSubView(faceOwners_, offset_, i);
}

const vectorVector& BoundaryMesh::faceCenters() const { return faceCenters_; }

View<const Vec3> BoundaryMesh::faceCenters(const localIdx i) const
{
    return extractSubView(faceCenters_, offset_, i);
}

const vectorVector& BoundaryMesh::ownerCellCenters() const { return ownerCellCenters_; }

View<const Vec3> BoundaryMesh::ownerCellCenters(const localIdx i) const
{
    return extractSubView(ownerCellCenters_, offset_, i);
}

const vectorVector& BoundaryMesh::faceNormals() const { return faceNormals_; }

View<const Vec3> BoundaryMesh::faceNormals(const localIdx i) const
{
    return extractSubView(faceNormals_, offset_, i);
}

const scalarVector& BoundaryMesh::faceAreas() const { return faceAreas_; }

View<const scalar> BoundaryMesh::faceAreas(const localIdx i) const
{
    return extractSubView(faceAreas_, offset_, i);
}

const vectorVector& BoundaryMesh::faceUnitNormals() const { return faceUnitNormals_; }

View<const Vec3> BoundaryMesh::faceUnitNormals(const localIdx i) const
{
    return extractSubView(faceUnitNormals_, offset_, i);
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

localIdx BoundaryMesh::neighbourRank(const localIdx i) const { return neighbourRank_[i]; }

const std::vector<localIdx>& BoundaryMesh::neighbourRank() const { return neighbourRank_; }

View<const scalar> BoundaryMesh::deltaCoeffs(const localIdx i) const
{
    return extractSubView(deltaCoeffs_, offset_, i);
}

localIdx BoundaryMesh::nProcBoundaryPatches() const { return procBoundaryPatches_; }

localIdx BoundaryMesh::nProcBoundaryFaces() const
{
    if (nProcBoundaryPatches() == 0)
    {
        return 0;
    }
    return offset_[offset_.size() - 1] - offset_[offset_.size() - nProcBoundaryPatches() - 1];
}

const std::vector<localIdx>& BoundaryMesh::offset() const { return offset_; }


} // namespace NeoN
