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

    using SizeType = size_t;

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
    const vectorVector& points() const { return points_; }
    vectorVector& points() { return points_; }

    /**
     * @brief Get the field of cell volumes in the mesh.
     *
     * @return The field of cell volumes in the mesh.
     */
    const scalarVector& cellVolumes() const { return cellVolumes_; }
    scalarVector& cellVolumes() { return cellVolumes_; }

    /**
     * @brief Get the field of cell centers in the mesh.
     *
     * @return The field of cell centers in the mesh.
     */
    const vectorVector& cellCenters() const { return cellCenters_; }
    vectorVector& cellCenters() { return cellCenters_; }

    /**
     * @brief Get the field of face centers.
     *
     * @return The field of face centers.
     */
    const vectorVector& faceCenters() const { return faceCenters_; }
    vectorVector& faceCenters() { return faceCenters_; }

    /**
     * @brief Get the field of face normal vectors.
     *
     * @return The field of face normal vectors.
     */
    const vectorVector& faceNormals() const { return faceNormals_; }
    vectorVector& faceNormals() { return faceNormals_; }

    /**
     * @brief Get the field of face areas.
     *
     * @return The field of face areas.
     */
    const scalarVector& faceAreas() const { return faceAreas_; }
    scalarVector& faceAreas() { return faceAreas_; }

    /**
     * @brief Get the list of labels of face owner cells.
     *
     * @return The list of labels of face owner cells.
     */
    const labelVector& faceOwners() const { return faceOwners_; }
    labelVector& faceOwners() { return faceOwners_; }

    /**
     * @brief Get the list of labels of face neighbor cells.
     *
     * @return The list of labels of face neighbor cells.
     */
    const labelVector& faceNeighbors() const { return faceNeighbors_; }
    labelVector& faceNeighbors() { return faceNeighbors_; }

    /**
     * @brief Get the number of cells in the mesh.
     *
     * @return The number of cells in the mesh.
     */
    SizeType nCells() const { return nCells_; }

    /**
     * @brief Get the number of internal faces in the mesh.
     *
     * @return The number of internal faces in the mesh.
     */
    SizeType nInternalFaces() const { return nInternalFaces_; }

    /**
     * @brief Get the number of boundary faces in the mesh.
     *
     * @return The number of boundary faces in the mesh.
     */
    SizeType nBoundaryFaces() const { return boundaryMesh_.nBoundaryFaces(); }

    /**
     * @brief Get the total number of faces including boundary and processor faces in the mesh.
     *
     * @return The  total number of faces in the mesh.
     */
    SizeType nTotalFaces() const
    {
        return nInternalFaces() + nBoundaryFaces() + nProcBoundaryFaces();
    }

    /**
     * @brief Get the number of boundaries patches in the mesh.
     *
     * @return The number of boundaries in the mesh.
     */
    SizeType nBoundaries() const { return boundaryMesh_.nBoundaries(); }

    /**
     * @brief Get the number of processor-boundary faces (inter-rank faces).
     *
     * Returns 0 for single-process runs.
     */
    SizeType nProcBoundaryFaces() const { return boundaryMesh_.nProcBoundaryFaces(); }

    /**
     * @brief the offset local cellIds for the global mesh.
     *
     * @return the global offset
     */
    localIdx globalOffset() const { return globalOffset_; }

    /**
     * @brief Get the boundary mesh.
     *
     * @return The boundary mesh.
     */
    const BoundaryMesh& boundaryMesh() const { return boundaryMesh_; }
    BoundaryMesh& boundaryMesh() { return boundaryMesh_; }

    /**
     * @brief Get the stencil data base.
     *
     * @return The stencil data base.
     */
    Dictionary& stencilDB() const { return stencilDataBase_; }

    /**
     * @brief Get the executor.
     *
     * @return The executor.
     */
    const Executor& exec() const { return exec_; }

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
    SizeType nCells_;

    /**
     * @brief Number of internal faces in the mesh.
     */
    SizeType nInternalFaces_;

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

/** @brief A factory function for a 2D uniform mesh (single-layer hex slab)
 *
 * Creates an nx × ny × 1 structured hex mesh on [0,Lx] × [0,Ly] × [0,1].
 * One cell thick in z, the standard quasi-2D finite-volume convention.
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


#ifdef NF_WITH_MPI_SUPPORT
/** @brief Factory function that builds the local 1D mesh partition for this MPI rank.
 *
 *  Creates a nCells-cell 1D mesh with regular boundary patches on the domain
 *  boundary and processor-boundary patches where the partition abuts another rank.
 *  Reads the current rank from the MPI environment.
 */
UnstructuredMesh create1DUniformMeshPart(const Executor exec, const localIdx nCells);

/** @brief Factory function that builds the local 2D mesh partition for a given rank.
 *
 *  Builds the rank-local part of a 2D uniform mesh that is partitioned into a
 *  @p ranksXPart × @p ranksYPart grid of square subdomains, where
 *  @p totalRanks == ranksXPart * ranksXPart and ranksXPart == ranksYPart.
 *  Each part is an nCells × nCells (one cell thick in z) hex slab.  Patches on
 *  the global domain boundary stay regular; patches shared with a neighbouring
 *  partition become processor-boundary patches.  Regular patches are stored
 *  before processor patches, matching create1DUniformMeshPart.
 *
 *  Unlike create1DUniformMeshPart the rank is passed explicitly so that a part
 *  can be built for any rank of the @p ranksXPart × @p ranksYPart grid.
 */
UnstructuredMesh create2DUniformMeshPart(
    const Executor exec, const localIdx nCells, const localIdx totalRanks, const localIdx rank
);

/** @brief Factory function that builds the local 2D mesh partition for a given rank,
 *  decomposing only along the x-axis.
 *
 *  Partitions the unit square into @p totalRanks × 1 strips along x.  Each rank
 *  owns an @p nCells × @p nCells (one cell thick in z) hex slab of width
 *  1/@p totalRanks.  The ymin and ymax patches are always regular; xmin is
 *  regular only on rank 0 and xmax only on the last rank — all other x-facing
 *  patches become processor-boundary patches.
 *
 *  Unlike create2DUniformMeshPart this variant accepts any @p totalRanks value,
 *  not just perfect squares.
 */
UnstructuredMesh create2DUniformMeshPartX(
    const Executor exec, const localIdx nCells, const localIdx totalRanks, const localIdx rank
);
#endif

} // namespace NeoN
