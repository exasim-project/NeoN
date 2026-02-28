// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/mesh/unstructured/io/meshConnectivity.hpp"
#include "NeoN/mesh/unstructured/io/meshGeometry.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

#include <vtkMultiBlockDataSet.h>
#include <vtkPartitionedDataSetCollection.h>
#include <vtkSmartPointer.h>

#include <vector>

class vtkUnstructuredGrid;
class vtkPartitionedDataSetCollection;

namespace NeoN::io
{

/// Build a vtkMultiBlockDataSet with volume grid and boundary patches.
///
/// Block 0 is "internalMesh" (vtkUnstructuredGrid with all volume cells).
/// Block 1 is "boundary" (nested vtkMultiBlockDataSet with named patches).
/// Patch names come from stencilDB "io::patchNames" if available.
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
