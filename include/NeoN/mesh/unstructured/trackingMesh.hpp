// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/segmentedVector.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN
{

/** @brief What happens to a point that leaves a cell through a face. */
enum class TrackingFaceType : label
{
    internal = 0, ///< go to the neighbour cell
    cyclic = 1,   ///< periodic: shift the position, continue in the partner cell
    escape = 2    ///< any other boundary: the point leaves the domain
};

/** @brief Read-only views of a TrackingMesh, captured by value inside kernels.
 *
 * The first block is the connectivity owned by TrackingMesh, the second block is taken
 * from the UnstructuredMesh it was built for (not stored twice).
 */
struct TrackingMeshView
{
    // owned by TrackingMesh
    SegmentedVectorView<const label, const localIdx> cellFaces;  ///< all faces of every cell
    SegmentedVectorView<const label, const localIdx> facePoints; ///< points of every face
    View<const label> faceBasePt;        ///< tet base point of every face (local index)
    View<const label> faceType;          ///< TrackingFaceType of every face
    View<const label> cyclicPartnerCell; ///< cyclic faces: cell behind the paired face
    View<const Vec3> cyclicShift;        ///< cyclic faces: translation to the paired face
    View<const Vec3> points;             ///< mesh points, on the executor

    // from UnstructuredMesh
    View<const Vec3> cellCenters;
    View<const scalar> cellVolumes;
    View<const Vec3> faceCenters;
    View<const Vec3> faceAreaVectors; ///< UnstructuredMesh::faceNormals(), i.e. Sf
    View<const label> faceOwners;
    View<const label> faceNeighbors; ///< internal faces only
    localIdx nInternalFaces;
};

/** @class TrackingMesh
 * @brief Connectivity for locating points (e.g. particles) in an UnstructuredMesh and
 * interpolating cell values to them, on the executor (CPU or GPU).
 *
 * UnstructuredMesh stores cell and face geometry but not which faces belong to a cell or
 * which points belong to a face. TrackingMesh adds exactly that, plus the tet base point
 * of every face (for the tet decomposition used by point interpolation) and the
 * periodic (cyclic) face pairing. It also keeps its own copy of the points on the
 * executor, since UnstructuredMesh may hold them on the host only.
 *
 * The geometry it does not own is read from the mesh, which must outlive it.
 */
class TrackingMesh
{
public:

    /** @brief Construct from a mesh and the connectivity it lacks.
     *
     * @param mesh              the mesh the connectivity belongs to
     * @param cellFaces         faces of every cell, internal and boundary
     * @param facePoints        points of every face, in face order
     * @param faceBasePt        tet base point of every face, local index in the face
     * @param faceType          TrackingFaceType of every face
     * @param cyclicPartnerCell cyclic faces: cell behind the paired face, -1 otherwise
     * @param cyclicShift       cyclic faces: translation to the paired face, 0 otherwise
     * @param points            mesh points on the executor of the mesh
     */
    TrackingMesh(
        const UnstructuredMesh& mesh,
        SegmentedVector<label, localIdx> cellFaces,
        SegmentedVector<label, localIdx> facePoints,
        Vector<label> faceBasePt,
        Vector<label> faceType,
        Vector<label> cyclicPartnerCell,
        Vector<Vec3> cyclicShift,
        Vector<Vec3> points
    )
        : mesh_(mesh), cellFaces_(std::move(cellFaces)), facePoints_(std::move(facePoints)),
          faceBasePt_(std::move(faceBasePt)), faceType_(std::move(faceType)),
          cyclicPartnerCell_(std::move(cyclicPartnerCell)), cyclicShift_(std::move(cyclicShift)),
          points_(std::move(points))
    {}

    /** @brief Views for use inside kernels. */
    [[nodiscard]] TrackingMeshView view() const
    {
        return {
            {cellFaces_.values().view(), cellFaces_.segments().view()},
            {facePoints_.values().view(), facePoints_.segments().view()},
            faceBasePt_.view(),
            faceType_.view(),
            cyclicPartnerCell_.view(),
            cyclicShift_.view(),
            points_.view(),
            mesh_.cellCenters().view(),
            mesh_.cellVolumes().view(),
            mesh_.faceCenters().view(),
            mesh_.faceNormals().view(),
            mesh_.faceOwners().view(),
            mesh_.faceNeighbors().view(),
            static_cast<localIdx>(mesh_.nInternalFaces())
        };
    }

    [[nodiscard]] const UnstructuredMesh& mesh() const { return mesh_; }

private:

    const UnstructuredMesh& mesh_;
    SegmentedVector<label, localIdx> cellFaces_;
    SegmentedVector<label, localIdx> facePoints_;
    Vector<label> faceBasePt_;
    Vector<label> faceType_;
    Vector<label> cyclicPartnerCell_;
    Vector<Vec3> cyclicShift_;
    Vector<Vec3> points_;
};

} // namespace NeoN
