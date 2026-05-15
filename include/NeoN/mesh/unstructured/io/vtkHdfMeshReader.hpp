// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::io
{

UnstructuredMesh readVtkHdf(const std::string& filePath, const Executor& exec);

} // namespace NeoN::io
