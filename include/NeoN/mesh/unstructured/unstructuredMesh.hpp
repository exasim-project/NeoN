// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "Kokkos_Sort.hpp"

#include "NeoN/core/dictionary.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/vector/vectorTypeDefs.hpp"
#include "NeoN/mesh/unstructured/boundaryMesh.hpp"

namespace NeoN
{

/**
 * @class UnstructuredMesh
 * @brief Represents an unstructured mesh in NeoN.
 *
 * The UnstructuredMesh class stores the data and provides access to the
 * properties of an unstructured mesh. It contains information such as mesh
 * points, cell volumes, cell centers, face areas, face centers, face owner
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
     * @param exec
     * @param points The field of mesh points.
     * @param cellVolumes The field of cell volumes in the mesh.
     * @param cellCenters The field of cell centers in the mesh.
     * @param faceNormals The field of face normal vectors.
     * @param faceCenters The field of face centers.
     * @param faceAreas The field of face areas.
     * @param faceOwners The list of labels of face owner cells.
     * @param faceNeighbors The list of labels of face neighbor cells.
     * @param boundaryMesh The boundary mesh.
     */
    UnstructuredMesh(
        Executor exec,
        vectorVector points,
        scalarVector cellVolumes,
        vectorVector cellCenters,
        vectorVector faceNormals,
        vectorVector faceCenters,
        scalarVector faceAreas,
        labelVector faceOwners,
        labelVector faceNeighbors,
        BoundaryMesh boundaryMesh
    );

    /**
     * @brief Constructor for the UnstructuredMesh class.
     * @note executor is determined from faceOwners
     *
     * @param points The field of mesh points.
     * @param cellVolumes The field of cell volumes in the mesh.
     * @param cellCenters The field of cell centers in the mesh.
     * @param faceNormals The field of face normal vectors.
     * @param faceCenters The field of face centers.
     * @param faceAreas The field of face areas.
     * @param faceOwners The list of labels of face owner cells.
     * @param faceNeighbors The list of labels of face neighbor cells.
     * @param boundaryMesh The boundary mesh.
     */
    UnstructuredMesh(
        vectorVector points,
        scalarVector cellVolumes,
        vectorVector cellCenters,
        vectorVector faceNormals,
        vectorVector faceCenters,
        scalarVector faceAreas,
        labelVector faceOwners,
        labelVector faceNeighbors,
        BoundaryMesh boundaryMesh
    );

    /**
     * @brief Get the field of mesh points.
     *
     * @return The field of mesh points.
     */
    const vectorVector& points() const;
    vectorVector& points();

    /**
     * @brief Get the field of cell volumes in the mesh.
     *
     * @return The field of cell volumes in the mesh.
     */
    const scalarVector& cellVolumes() const;
    scalarVector& cellVolumes();

    /**
     * @brief Get the field of cell centers in the mesh.
     *
     * @return The field of cell centers in the mesh.
     */
    const vectorVector& cellCenters() const;
    vectorVector& cellCenters();

    /**
     * @brief Get the field of face centers.
     *
     * @return The field of face centers.
     */
    const vectorVector& faceCenters() const;
    vectorVector& faceCenters();

    /**
     * @brief Get the field of face normal vectors.
     *
     * @return The field of face normal vectors.
     */
    const vectorVector& faceNormals() const;
    vectorVector& faceNormals();

    /**
     * @brief Get the field of face areas.
     *
     * @return The field of face areas.
     */
    const scalarVector& faceAreas() const;
    scalarVector& faceAreas();

    /**
     * @brief Get the list of labels of face owner cells.
     *
     * @return The list of labels of face owner cells.
     */
    const labelVector& faceOwners() const;
    labelVector& faceOwners();

    /**
     * @brief Get the list of labels of face neighbor cells.
     *
     * @return The list of labels of face neighbor cells.
     */
    const labelVector& faceNeighbors() const;
    labelVector& faceNeighbors();

    /**
     * @brief Get the number of cells in the mesh.
     *
     * @return The number of cells in the mesh.
     */
    localIdx nCells() const;

    /**
     * @brief Get the number of internal faces in the mesh.
     *
     * @return The number of internal faces in the mesh.
     */
    localIdx nInternalFaces() const;

    /**
     * @brief Get the number of boundary faces in the mesh.
     *
     * @return The number of boundary faces in the mesh.
     */
    localIdx nBoundaryFaces() const;

    localIdx nProcBoundaryFaces() const;

    /**
     * @brief Get the total number of faces including boundary and processor faces in the mesh.
     *
     * @return The  total number of faces in the mesh.
     */
    localIdx nTotalFaces() const;

    /**
     * @brief Get the number of boundaries patches in the mesh.
     *
     * @return The number of boundaries in the mesh.
     */
    localIdx nBoundaries() const;

    /**
     * @brief the offset local cellIds for the global mesh.
     *
     * @return the global offset
     */
    localIdx globalOffset() const;

    /**
     * @brief Get the boundary mesh.
     *
     * @return The boundary mesh.
     */
    const BoundaryMesh& boundaryMesh() const;

    BoundaryMesh& boundaryMesh();

    /**
     * @brief Get the stencil data base.
     *
     * @return The stencil data base.
     */
    Dictionary& stencilDB() const;

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
     * @brief Vector of cell centers in the mesh.
     */
    vectorVector cellCenters_;

    /**
     * @brief Vector of face normal vectors.
     *
     * The face normal vectors are defined as the normal vector to the face
     * with magnitude equal to the face area.
     */
    vectorVector faceNormals_;

    /**
     * @brief Vector of face centers.
     */
    vectorVector faceCenters_;

    /**
     * @brief Vector of face areas.
     */
    scalarVector faceAreas_;

    /**
     * @brief Vector of labels of face owner cells.
     */
    labelVector faceOwners_;

    /**
     * @brief Vector of labels of face neighbor cells.
     */
    labelVector faceNeighbors_;

    /**
     * @brief Number of cells in the mesh.
     */
    localIdx nCells_;

    /**
     * @brief Number of internal faces in the mesh.
     */
    localIdx nInternalFaces_;

    /**
     * @brief Boundary mesh.
     *
     * The boundary mesh is a collection of boundary patches
     * that are used to define boundary conditions in the mesh.
     */
    BoundaryMesh boundaryMesh_;

    //
    localIdx globalOffset_;

    /**
     * @brief Stencil data base.
     *
     * The stencil data base is used to register stencils.
     */
    mutable Dictionary stencilDataBase_;
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

/** @brief A factory function for a 1D uniform mesh
 *
 * A 1D mesh in 3D space in which each cell has a left and a right boundary face.
 * The 1D mesh is aligned with the x coordinate of Cartesian coordinate system.
 */
UnstructuredMesh create1DUniformMesh(const Executor exec, const localIdx nCells, scalar lx = 1.0);

/** @brief A factory function for a 2D uniform mesh (OpenFOAM-style hex slab)
 *
 * Creates an nx × ny × 1 structured hex mesh on [0,Lx] × [0,Ly] × [0,1].
 * One cell thick in z, like OpenFOAM 2D meshes.
 * Four boundary patches: left (x=0), right (x=Lx), bottom (y=0), top (y=Ly)
 */
UnstructuredMesh create2DUniformMesh(
    const Executor exec, localIdx nx, localIdx ny, scalar lx = 1.0, scalar ly = 1.0
);

/** @brief A factory function for a uniform 3D hex mesh
 *
 * Creates an nx × ny × nz structured hex mesh on [0,Lx] × [0,Ly] × [0,Lz].
 * Six boundary patches: left (x=0), right (x=Lx), bottom (y=0), top (y=Ly),
 * front (z=0), back (z=Lz).
 */
UnstructuredMesh create3DUniformMesh(
    const Executor exec,
    localIdx nx,
    localIdx ny,
    localIdx nz,
    scalar lx = 1.0,
    scalar ly = 1.0,
    scalar lz = 1.0
);

} // namespace NeoN
