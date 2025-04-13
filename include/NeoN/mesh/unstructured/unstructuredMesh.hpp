// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#pragma once

#include "Kokkos_Sort.hpp"

#include "NeoN/fields/fieldTypeDefs.hpp"
#include "NeoN/mesh/unstructured/boundaryMesh.hpp"
#include "NeoN/finiteVolume/cellCentred/stencil/stencilDataBase.hpp"

namespace NeoN
{

/**
 * @class UnstructuredMesh
 * @brief Represents an unstructured mesh in NeoN.
 *
 * The UnstructuredMesh class stores the data and provides access to the
 * properties of an unstructured mesh. It contains information such as mesh
 * points, cell volumes, cell centres, face areas, face centres, face owner
 * cells, face neighbour cells, and boundary information. It also provides
 * methods to retrieve the number of cells, internal faces, boundary faces,
 * boundaries, and faces in the mesh. Additionally, it includes a boundary mesh
 * and a stencil data base. The executor is used to run parallel operations on
 * the mesh.
 */
class UnstructuredMesh
{
public:

    /**
     * @brief Constructor for the UnstructuredMesh class.
     *
     * @param points The field of mesh points.
     * @param cellVolumes The field of cell volumes in the mesh.
     * @param cellCentres The field of cell centres in the mesh.
     * @param faceAreas The field of area face normals.
     * @param faceCentres The field of face centres.
     * @param magFaceAreas The field of magnitudes of face areas.
     * @param faceOwner The field of face owner cells.
     * @param faceNeighbour The field of face neighbour cells.
     * @param nCells The number of cells in the mesh.
     * @param nInternalFaces The number of internal faces in the mesh.
     * @param nBoundaryFaces The number of boundary faces in the mesh.
     * @param nBoundaries The number of boundaries in the mesh.
     * @param nFaces The number of faces in the mesh.
     * @param boundaryMesh The boundary mesh.
     */
    UnstructuredMesh(
        vectorVector points,
        scalarVector cellVolumes,
        vectorVector cellCentres,
        vectorVector faceAreas,
        vectorVector faceCentres,
        scalarVector magFaceAreas,
        labelVector faceOwner,
        labelVector faceNeighbour,
        size_t nCells,
        size_t nInternalFaces,
        size_t nBoundaryFaces,
        size_t nBoundaries,
        size_t nFaces,
        BoundaryMesh boundaryMesh
    );

    /**
     * @brief Get the field of mesh points.
     *
     * @return The field of mesh points.
     */
    const vectorVector& points() const;

    /**
     * @brief Get the field of cell volumes in the mesh.
     *
     * @return The field of cell volumes in the mesh.
     */
    const scalarVector& cellVolumes() const;

    /**
     * @brief Get the field of cell centres in the mesh.
     *
     * @return The field of cell centres in the mesh.
     */
    const vectorVector& cellCentres() const;

    /**
     * @brief Get the field of face centres.
     *
     * @return The field of face centres.
     */
    const vectorVector& faceCentres() const;

    /**
     * @brief Get the field of area face normals.
     *
     * @return The field of area face normals.
     */
    const vectorVector& faceAreas() const;

    /**
     * @brief Get the field of magnitudes of face areas.
     *
     * @return The field of magnitudes of face areas.
     */
    const scalarVector& magFaceAreas() const;

    /**
     * @brief Get the field of face owner cells.
     *
     * @return The field of face owner cells.
     */
    const labelVector& faceOwner() const;

    /**
     * @brief Get the field of face neighbour cells.
     *
     * @return The field of face neighbour cells.
     */
    const labelVector& faceNeighbour() const;

    /**
     * @brief Get the number of cells in the mesh.
     *
     * @return The number of cells in the mesh.
     */
    size_t nCells() const;

    /**
     * @brief Get the number of internal faces in the mesh.
     *
     * @return The number of internal faces in the mesh.
     */
    size_t nInternalFaces() const;

    /**
     * @brief Get the number of boundary faces in the mesh.
     *
     * @return The number of boundary faces in the mesh.
     */
    size_t nBoundaryFaces() const;

    /**
     * @brief Get the number of boundaries in the mesh.
     *
     * @return The number of boundaries in the mesh.
     */
    size_t nBoundaries() const;

    /**
     * @brief Get the number of faces in the mesh.
     *
     * @return The number of faces in the mesh.
     */
    size_t nFaces() const;

    /**
     * @brief Get the boundary mesh.
     *
     * @return The boundary mesh.
     */
    const BoundaryMesh& boundaryMesh() const;

    /**
     * @brief Get the stencil data base.
     *
     * @return The stencil data base.
     */
    StencilDataBase& stencilDB() const;

    /**
     * @brief Get the executor.
     *
     * @return The executor.
     */
    const Executor& exec() const;

private:

    /**
     * @brief Executor
     *
     * The executor is used to run parallel operations on the mesh.
     */
    const Executor exec_;

    /**
     * @brief Vector of mesh points.
     */
    vectorVector points_;

    /**
     * @brief Vector of cell volumes in the mesh.
     */
    scalarVector cellVolumes_;

    /**
     * @brief Vector of cell centres in the mesh.
     */
    vectorVector cellCentres_;

    /**
     * @brief Vector of area face normals.
     *
     * The area face normals are defined as the normal vector to the face
     * with magnitude equal to the face area.
     */
    vectorVector faceAreas_;

    /**
     * @brief Vector of face centres.
     */
    vectorVector faceCentres_;

    /**
     * @brief Vector of magnitudes of face areas.
     */
    scalarVector magFaceAreas_;

    /**
     * @brief Vector of face owner cells.
     */
    labelVector faceOwner_;

    /**
     * @brief Vector of face neighbour cells.
     */
    labelVector faceNeighbour_;

    /**
     * @brief Number of cells in the mesh.
     */
    size_t nCells_;

    /**
     * @brief Number of internal faces in the mesh.
     */
    size_t nInternalFaces_;

    /**
     * @brief Number of boundary faces in the mesh.
     */
    size_t nBoundaryFaces_;

    /**
     * @brief Number of boundaries in the mesh.
     */
    size_t nBoundaries_;

    /**
     * @brief Number of faces in the mesh.
     */
    size_t nFaces_;

    /**
     * @brief Boundary mesh.
     *
     * The boundary mesh is a collection of boundary patches
     * that are used to define boundary conditions in the mesh.
     */
    BoundaryMesh boundaryMesh_;

    /**
     * @brief Stencil data base.
     *
     * The stencil data base is used to register stencils.
     */
    mutable StencilDataBase stencilDataBase_;
};

/** @brief creates a mesh containing only a single cell
 * @warn currently this is only a 2D mesh
 *
 * a 2D mesh in 3D space with left, right, top, bottom boundary faces
 * with the centre at (0.5, 0.5, 0.0)
 * left, top, right, bottom faces
 * and four boundaries one left, right, top, bottom
 */
UnstructuredMesh createSingleCellMesh(const Executor exec);

/** @brief A factory function for a 1D mesh
 *
 * A 1D mesh in 3D space in which each cell has a left and a right face.
 * The 1D mesh is aligned with the x coordinate of Cartesian coordinate system.
 */
UnstructuredMesh create1DUniformMesh(const Executor exec, const size_t nCells);


} // namespace NeoN
