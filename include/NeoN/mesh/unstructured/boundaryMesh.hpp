// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/vector/vectorTypeDefs.hpp"

namespace NeoN
{

/**
 * @class BoundaryMesh
 * @brief Represents boundaries of an unstructured mesh.
 *
 * The BoundaryMesh class stores information about the boundary faces and their
 * properties in an unstructured mesh. It provides access to various fields such
 * as face cells, face centers, face normals, face areas normals, magnitudes of
 * face areas normals, delta vectors, weights, delta coefficients, and offsets.
 *
 * Boundary faces are organised such that regular boundaries are stored first, followed
 * by the processor boundaries.
 *
 * The class also provides getter methods to access the individual fields and
 * their components.
 *
 * @tparam Executor The type of the executor used for computations.
 */
class BoundaryMesh
{
public:

    void validate() const;

    /**
     * @brief Constructor for the BoundaryMesh class.
     *
     * @param exec The executor used for computations.
     * @param faceOwners A list of labels of owner cells of boundary faces.
     * @param faceCenters A field of face centers.
     * @param ownerCellCenters A field of centers of owner cells of boundary faces.
     * @param faceNormals A field of face normal vectors.
     * @param faceAreas A field of face areas.
     * @param faceUnitNormals A field of face unit normal vectors.
     * @param delta A field of delta vectors.
     * @param weights A field of weights used in cell to face interpolation.
     * @param deltaCoeffs A field of the inverse of distances between boundary faces and their
     * neighboring cell centers
     * @param offset The offsets of the boundary faces, i.e. offset[i], offset[i+1] define beginning
     * and end of a patch.
     * @param neighbourRank corresponding mpiRank ouf neighbour
     */
    BoundaryMesh(
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
    );


    /**
     * @brief Get the list of labels of owner cells of boundary faces.
     *
     * @return A constant reference to the field of owner cells.
     */
    const labelVector& faceOwners() const;

    // TODO either dont mix return types, ie dont use view and Vector
    // for functions with same name
    /**
     * @brief Get a view of labels of owner cells for a specific boundary face.
     *
     * @param i The index of the boundary face.
     * @return A view of labels of owner cells for the specified boundary face.
     */
    View<const label> faceOwners(const localIdx i) const;

    /**
     * @brief Get the field of face centers.
     *
     * @return A constant reference to the field of face centers.
     */
    const vectorVector& faceCenters() const;

    /**
     * @brief Get a view of face centers for a specific boundary face.
     *
     * @param i The index of the boundary face.
     * @return A view of face centers for the specified boundary face.
     */
    View<const Vec3> faceCenters(const localIdx i) const;

    /**
     * @brief Get the field of centers of owner cells of boundary faces.
     *
     * @return A constant reference to the field of owner cell centers.
     */
    const vectorVector& ownerCellCenters() const;

    /**
     * @brief Get a view of owner cell centers for a specific boundary face.
     *
     * @param i The index of the boundary face.
     * @return A view of owner cell centers for the specified boundary face.
     */
    View<const Vec3> ownerCellCenters(const localIdx i) const;

    /**
     * @brief Get the field of face areas normals.
     *
     * @return A constant reference to the field of face areas normals.
     */
    const vectorVector& faceNormals() const;

    /**
     * @brief Get a view of face areas normals for a specific boundary face.
     *
     * @param i The index of the boundary face.
     * @return A view of face areas normals for the specified boundary face.
     */
    View<const Vec3> faceNormals(const localIdx i) const;

    /**
     * @brief Get the field of face areas.
     *
     * @return A constant reference to the field of face areas.
     */
    const scalarVector& faceAreas() const;

    /**
     * @brief Get a view of face areas for a specific boundary face.
     *
     * @param i The index of the boundary face.
     * @return A view of face areas for the specified boundary face.
     */
    View<const scalar> faceAreas(const localIdx i) const;

    /**
     * @brief Get the field of face unit normal vectors.
     *
     * @return A constant reference to the field of face unit normal vectors.
     */
    const vectorVector& faceUnitNormals() const;

    /**
     * @brief Get a view of face unit normal vectors for a specific boundary face.
     *
     * @param i The index of the boundary face.
     * @return A view of face unit normal vectors for the specified boundary face.
     */
    View<const Vec3> faceUnitNormals(const localIdx i) const;

    /**
     * @brief Get the field of delta vectors.
     *
     * @return A constant reference to the field of delta vectors.
     */
    const vectorVector& delta() const;

    /**
     * @brief Get a view of delta vectors for a specific boundary face.
     *
     * @param i The index of the boundary face.
     * @return A view of delta vectors for the specified boundary face.
     */
    View<const Vec3> delta(const localIdx i) const;

    /**
     * @brief Get the field of weights.
     *
     * @return A constant reference to the boundary field of weights.
     */
    const scalarVector& weights() const;

    /**
     * @brief Get a view of weights for a specific boundary face.
     *
     * @param i The index of the boundary face.
     * @return A view of weights for the specified boundary face.
     */
    View<const scalar> weights(const localIdx i) const;

    /**
     * @brief Get the field of delta coefficients.
     *
     * @return A constant reference to the field of delta coefficients.
     */
    const scalarVector& deltaCoeffs() const;

    /**
     * @brief Get a view of delta coefficients for a specific boundary face.
     *
     * @param i The index of the boundary face.
     * @return A view of delta coefficients for the specified boundary face.
     */
    View<const scalar> deltaCoeffs(const localIdx i) const;

    /**
     * @brief Given a patchId the corresponding neighbour Rank gets returned
     *
     * -1 for patches without a distributed neighbour
     */
    localIdx neighbourRank(const localIdx i) const;

    /**
     * @brief Get the offset of the boundary faces.
     *
     * @return A constant reference to the offset of the boundary faces.
     */
    // TODO consistent use of Vector on CPU
    const std::vector<localIdx>& offset() const;

    /**@brief number of proc boundary patches */
    localIdx nProcBoundaryPatches() const;

    /**
     * @brief Get the offset of the boundary faces.
     *
     * @return A constant reference to the offset of the boundary faces.
     */
    const std::vector<localIdx>& neighbourRank() const;

    /**
     * @brief Returns the neighbour rank for the processor patch whose face range matches the
     * given range.
     *
     * @param range The [start, end) face index range of the processor patch.
     * @return The MPI rank of the neighbour for that patch.
     */
    localIdx neighbourRankForRange(std::pair<localIdx, localIdx> range) const;

    /**
     * @brief Get the number of the boundaries.
     */
    localIdx nBoundaries() const { return offset_.size() - 1; }

    /**
     * @brief Get the number of the boundary faces.
     */
    localIdx nBoundaryFaces() const { return faceOwners_.size() - nProcBoundaryFaces(); }

    /**
     * @brief Get the number of the processor boundary faces.
     */
    localIdx nProcBoundaryFaces() const;

    /**
     * @brief Return whether this local mesh is a part of a global distributed mesh
     */
    bool isDistributed() const { return nProcBoundaryFaces() > 0; }

private:

    /**
     * @brief Executor used for computations.
     */
    const Executor exec_;

    /**
     * @brief Vector of labels of face owner cells.
     *
     * A list of labels of the owner cells of boundary faces.
     */
    labelVector faceOwners_;

    /**
     * @brief Vector of face centers.
     */
    vectorVector faceCenters_;

    /**
     * @brief Vector of centers of owner cells of boundary faces.
     */
    vectorVector ownerCellCenters_;

    /**
     * @brief Vector of face areas normals.
     */
    vectorVector faceNormals_;

    /**
     * @brief Vector of face areas.
     */
    scalarVector faceAreas_;

    /**
     * @brief Vector of face unit normal vectors.
     */
    vectorVector faceUnitNormals_;

    /**
     * @brief Vector of delta vectors.
     *
     * The delta vector is defined as the vector from the face centre to the
     * cell centre.
     */
    vectorVector delta_;

    /**
     * @brief Vector of weights.
     *
     * The weights are used in cell to face interpolation.
     */
    scalarVector weights_;

    /**
     * @brief Vector of delta coefficients.
     *
     * Vector of cell to face distances.
     */
    scalarVector deltaCoeffs_;

    /**
     * @brief Offset of the boundary faces.
     *
     * The offset is used to access the boundary faces of each boundary.
     */
    // TODO consistent use of Vector on CPU
    std::vector<localIdx> offset_;

    /**
     * @brief number of processor patches
     *
     * Vector of cell to face distances.
     */
    localIdx procBoundaryPatches_;

    /**
     * @brief The mpi rank of the corresponding neighbour patch
     */
    std::vector<localIdx> neighbourRank_;
};

} // namespace NeoN
