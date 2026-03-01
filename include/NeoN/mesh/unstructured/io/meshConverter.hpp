// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/mesh/unstructured/io/meshConnectivity.hpp"
#include "NeoN/mesh/unstructured/io/meshGeometry.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/mesh/unstructured/boundaryMesh.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

#include <vtkMultiBlockDataSet.h>
#include <vtkPartitionedDataSetCollection.h>
#include <vtkSmartPointer.h>

#include <string_view>
#include <vector>

class vtkUnstructuredGrid;
class vtkPartitionedDataSetCollection;

namespace NeoN::io
{

/// StencilDB key for face-node connectivity (SegmentedVector<localIdx,localIdx>).
constexpr std::string_view stencilFaceNodes = "io::faceNodes";

/// StencilDB key for patch names (std::vector<std::string>).
constexpr std::string_view stencilPatchNames = "io::patchNames";

/// Build BoundaryMesh from geometry arrays computed on host (SerialExecutor).
///
/// Constructs per-boundary-face quantities (normals, deltas, weights, etc.)
/// and returns a BoundaryMesh on the target executor.
BoundaryMesh buildBoundaryMesh(
    const Executor& exec,
    const Vector<localIdx>& faceOwner,
    const Vector<Vec3>& faceCentres,
    const Vector<Vec3>& cellCentres,
    const Vector<Vec3>& faceAreas,
    const Vector<scalar>& magFaceAreas,
    localIdx nInternalFaces,
    localIdx nBoundaryFaces,
    const std::vector<localIdx>& patchOffsets
);

/// Build a vtkMultiBlockDataSet with volume grid and boundary patches.
///
/// Block 0 is "internalMesh" (vtkUnstructuredGrid with all volume cells).
/// Block 1 is "boundary" (nested vtkMultiBlockDataSet with named patches).
/// Patch names come from stencilDB stencilPatchNames if available.
vtkSmartPointer<vtkMultiBlockDataSet> buildMultiBlockMesh(const UnstructuredMesh& mesh);

/// Build a vtkPartitionedDataSetCollection with vtkDataAssembly.
///
/// Dataset 0 is "internalMesh" (vtkUnstructuredGrid with all volume cells).
/// Datasets 1..N are named boundary patches (vtkPolyData with boundary faces).
/// The assembly hierarchy is: Root/internalMesh, Root/boundary/{patchName, ...}.
vtkSmartPointer<vtkPartitionedDataSetCollection> buildPartitionedMesh(const UnstructuredMesh& mesh);

/// Extract patch names from a multiblock boundary sub-block.
///
/// Must be called from the same library that set the metadata (i.e. NeoN)
/// because VTK information keys are file-static and can differ across
/// translation units when VTK is statically linked into a shared library.
std::vector<std::string> multiBlockPatchNames(vtkMultiBlockDataSet* boundary);


} // namespace NeoN::io
