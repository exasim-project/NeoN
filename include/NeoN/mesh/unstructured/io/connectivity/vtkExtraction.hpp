// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/mesh/unstructured/io/connectivity/types.hpp"

class vtkUnstructuredGrid;

namespace NeoN::io
{

/// Extract cell connectivity from a VTK unstructured grid.
/// Results are placed on exec (default: SerialExecutor).
CellConnectivity
extractCellConnectivity(vtkUnstructuredGrid* grid, const Executor& exec = SerialExecutor {});

} // namespace NeoN::io
