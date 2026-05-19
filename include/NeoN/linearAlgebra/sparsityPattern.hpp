// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

// This header is kept for backwards compatibility.
// The types have been moved to separate, more focused headers.
#include "NeoN/linearAlgebra/sparsityView.hpp"
#include "NeoN/linearAlgebra/csrSparsityPattern.hpp"
#include "NeoN/linearAlgebra/cooSparsityPattern.hpp"

namespace NeoN::la
{

// SparsityPattern is an alias for the CSR sparsity pattern.
template<typename IndexType>
using SparsityPattern = CsrSparsityPattern<IndexType>;

} // namespace NeoN::la
