// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::io
{

void writeVtkHdf(const UnstructuredMesh& mesh, const std::string& filePath);

} // namespace NeoN::io
