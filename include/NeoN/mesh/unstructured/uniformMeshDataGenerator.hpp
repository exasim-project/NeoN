// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

#include "NeoN/core/primitives/vec3.hpp"

#include <vector>

namespace NeoN::detail
{
/** @brief Parameters describing a 3D uniform Cartesian mesh
 *
 * Defines the number of cells in each direction and the physical domain size.
 * Provides helper functions for indexing cells and mesh points in a structured layout.
 */
struct MeshParams
{
    localIdx nx, ny, nz;
    scalar Lx, Ly, Lz;

    // Derived spacings
    scalar dx() const { return Lx / static_cast<scalar>(nx); }
    scalar dy() const { return Ly / static_cast<scalar>(ny); }
    scalar dz() const { return Lz / static_cast<scalar>(nz); }

    // Layout of cells in memory: go through x first, then y, then z.
    // So cell indices in the first layer are ordered as:
    // (0,0,0), (1,0,0), ..., (nx-1,0,0),
    // (0,1,0), (1,1,0), ..., (nx-1,1,0),
    // ...,
    // (0,ny-1,0), (1,ny-1,0), ..., (nx-1,ny-1,0)
    localIdx cellIdx(localIdx i, localIdx j, localIdx k) const { return k * nx * ny + j * nx + i; }

    localIdx ptIdx(localIdx i, localIdx j, localIdx k) const
    {
        return k * (nx + 1) * (ny + 1) + j * (nx + 1) + i;
    }
};

/** @brief Cell geometric data for a uniform mesh
 *
 * Contains precomputed cell volumes and cell-center coordinates
 * for a structured Cartesian mesh.
 */
struct CellData
{
    std::vector<scalar> volumes;
    std::vector<Vec3> centers;
};

/** @brief Face geometric and topological data for internal mesh faces
 *
 * Stores face areas, centroids, magnitudes, and owner-neighbour connectivity
 * for all internal faces of a structured 3D uniform mesh.
 */
struct FaceData
{
    std::vector<Vec3> areas;
    std::vector<Vec3> centers;
    std::vector<scalar> magnitudes;
    std::vector<label> owner;
    std::vector<label> neighbour;
};

/** @brief Generate mesh vertex coordinates for a uniform Cartesian grid
 *
 * Creates a structured set of points spanning a 3D rectangular domain
 * with uniform spacing in x, y, and z directions.
 *
 * @param p Mesh parameters defining resolution and domain size
 * @return Vector of point coordinates in structured ordering
 */
std::vector<Vec3> generatePoints(const MeshParams& p);

/** @brief Generate cell volumes and cell-center coordinates
 *
 * Computes geometric properties of each cell in a uniform Cartesian mesh,
 * assuming constant spacing in all coordinate directions.
 *
 * @param p Mesh parameters
 * @return CellData containing volumes and cell centers
 */
CellData generateCellData(const MeshParams& p);

/** @brief Generate internal face data for a 3D uniform mesh
 *
 * Constructs face areas, centers, magnitudes, and owner/neighbour connectivity
 * for all internal faces in x-, y-, and z-directions.
 *
 * Face ordering:
 * - First: x-normal faces
 * - Second: y-normal faces
 * - Third: z-normal faces
 *
 * @param p Mesh parameters
 * @param nInternalFaces Number of internal faces
 * @param nFaces Total number of faces
 * @return FaceData containing internal face geometry and connectivity
 */
FaceData
generateInternalFaces(const MeshParams& p, const localIdx nInternalFaces, const localIdx nFaces);

/** @brief Generate boundary face data for a 3D uniform mesh
 *
 * Constructs boundary face geometry, connectivity, and interpolation data
 * for all six domain boundaries (x-, y-, z-directions).
 *
 * Supports 1D, 2D, and 3D configurations depending on dimension parameter.
 *
 * @param exec Execution context
 * @param dim Spatial dimension (1, 2, or 3)
 * @param p Mesh parameters
 * @param centers Cell center coordinates
 * @param nInternalFaces Number of internal faces
 * @param nBoundaryFaces Number of boundary faces
 * @param offset Boundary face offset mapping for patch indexing
 * @param faces face data including boundary faces
 * @return BoundaryMesh containing all boundary-related geometric data
 */
BoundaryMesh generateBoundaryData(
    const Executor exec,
    const int dim,
    const MeshParams& p,
    const std::vector<Vec3>& centers,
    const localIdx nInternalFaces,
    const localIdx nBoundaryFaces,
    const std::vector<localIdx> offset,
    FaceData& faces
);
} // namespace NeoN::detail
