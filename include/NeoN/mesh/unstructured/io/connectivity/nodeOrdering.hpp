// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/mesh/unstructured/io/connectivity/types.hpp"

namespace NeoN::io
{

/** Order quad nodes: returns [n0, n1, n2, n3] in connected edge order (0-based). */
std::vector<localIdx> orderQuadNodes(const CellInfo& cell);

/** Order tet nodes: returns [base0, base1, base2, apex] (0-based). */
std::vector<localIdx> orderTetNodes(const CellInfo& cell);

/** Order hex nodes: returns [bottom0..3, top0..3] (0-based). */
std::vector<localIdx> orderHexNodes(const CellInfo& cell);

/** Order pyramid nodes: returns [base0..3, apex] (0-based). */
std::vector<localIdx> orderPyramidNodes(const CellInfo& cell);

/** Order wedge/prism nodes: returns [bottom0..2, top0..2] (0-based). */
std::vector<localIdx> orderWedgeNodes(const CellInfo& cell);

} // namespace NeoN::io
