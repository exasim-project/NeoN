// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/macros.hpp"
#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/array.hpp"
#include "NeoN/linearAlgebra/cooSparsityPattern.hpp"
#include "NeoN/linearAlgebra/utilities.hpp"

namespace NeoN::la
{

template<typename IndexType>
void CooSparsityPattern<IndexType>::validate() const
{
    NF_ASSERT(rowIdxs_.exec() == colIdxs_.exec(), "Executors are not the same");
    NF_ASSERT(rowIdxs_.size() == colIdxs_.size(), "COO size mismatch");
}

template<typename IndexType>
CooSparsityPattern<IndexType>::CooSparsityPattern(
    Vector<IndexType>&& colIdx, Vector<IndexType>&& rowIdx, Dimensions dim
)
    : dimensions_(dim), rowIdxs_(std::move(rowIdx)), colIdxs_(std::move(colIdx)),
      rowOffs_(rowsToRowOffs(rowIdxs_))
{
    validate();
}

template<typename IndexType>
CooSparsityPattern<IndexType>::CooSparsityPattern(const CooSparsityPattern& sp)
    : dimensions_(sp.dimensions_), rowIdxs_(sp.rowIdxs_), rowOffs_(sp.rowOffs_),
      colIdxs_(sp.colIdxs_)
{}

#define NN_DECLARE_SPARSITY(TYPENAME) template class CooSparsityPattern<TYPENAME>

NN_FOR_ALL_INTEGER_TYPES(NN_DECLARE_SPARSITY);

} // namespace NeoN::la
