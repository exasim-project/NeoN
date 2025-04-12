// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#pragma once

#include "Kokkos_Sort.hpp"

#include "NeoN/fields/fieldTypeDefs.hpp"
#include "NeoN/mesh/unstructured/boundaryMesh.hpp"
#include "NeoN/finiteVolume/cellCentred/stencil/stencilDataBase.hpp"

#include <vector>
#include <numeric>
#include <algorithm>

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
        vectorField points,
        scalarField cellVolumes,
        vectorField cellCentres,
        vectorField faceAreas,
        vectorField faceCentres,
        scalarField magFaceAreas,
        labelField faceOwner,
        labelField faceNeighbour,
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
    const vectorField& points() const;

    /**
     * @brief Get the field of cell volumes in the mesh.
     *
     * @return The field of cell volumes in the mesh.
     */
    const scalarField& cellVolumes() const;

    /**
     * @brief Get the field of cell centres in the mesh.
     *
     * @return The field of cell centres in the mesh.
     */
    const vectorField& cellCentres() const;

    /**
     * @brief Get the field of face centres.
     *
     * @return The field of face centres.
     */
    const vectorField& faceCentres() const;

    /**
     * @brief Get the field of area face normals.
     *
     * @return The field of area face normals.
     */
    const vectorField& faceAreas() const;

    /**
     * @brief Get the field of magnitudes of face areas.
     *
     * @return The field of magnitudes of face areas.
     */
    const scalarField& magFaceAreas() const;

    /**
     * @brief Get the field of face owner cells.
     *
     * @return The field of face owner cells.
     */
    const labelField& faceOwner() const;

    /**
     * @brief Get the field of face neighbour cells.
     *
     * @return The field of face neighbour cells.
     */
    const labelField& faceNeighbour() const;

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
     * @brief Field of mesh points.
     */
    vectorField points_;

    /**
     * @brief Field of cell volumes in the mesh.
     */
    scalarField cellVolumes_;

    /**
     * @brief Field of cell centres in the mesh.
     */
    vectorField cellCentres_;

    /**
     * @brief Field of area face normals.
     *
     * The area face normals are defined as the normal vector to the face
     * with magnitude equal to the face area.
     */
    vectorField faceAreas_;

    /**
     * @brief Field of face centres.
     */
    vectorField faceCentres_;

    /**
     * @brief Field of magnitudes of face areas.
     */
    scalarField magFaceAreas_;

    /**
     * @brief Field of face owner cells.
     */
    labelField faceOwner_;

    /**
     * @brief Field of face neighbour cells.
     */
    labelField faceNeighbour_;

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
