// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/mesh/unstructured/io/connectivity/types.hpp"

namespace NeoN::io
{

/// Build face topology from cell-to-node connectivity.
///
/// Each cell's faces are identified using element-type face templates,
/// then deduplicated using canonical (sorted) face keys. Shared faces
/// become internal faces; unshared faces become boundary faces.
/// The deduplication runs on the host; results are copied to exec.
FaceTopology buildFaceTopology(const Executor& exec, const CellConnectivity& connectivity);

} // namespace NeoN::io
