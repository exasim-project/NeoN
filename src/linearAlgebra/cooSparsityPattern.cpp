// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/macros.hpp"
#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/array.hpp"
#include "NeoN/linearAlgebra/cooSparsityPattern.hpp"

namespace NeoN::la
{

template<typename IndexType>
void CooSparsityPattern<IndexType>::validate() const
{
    NF_ASSERT(rowOffs_.exec() == colIdxs_.exec(), "Executors are not the same");
    NF_ASSERT(rowOffs_.size() == colIdxs_.size(), "COO size mismatch");
}

template<typename IndexType>
CooSparsityPattern<IndexType>::CooSparsityPattern(
    Vector<IndexType>&& colIdx, Vector<IndexType>&& rowOffs, Dimensions dim
)
    : dimensions_(dim), rowOffs_(std::move(rowOffs)), colIdxs_(std::move(colIdx)),
      rowOffsV_(rowOffs_.view()), colIdxsV_(colIdxs_.view())
{
    validate();
}

template<typename IndexType>
CooSparsityPattern<IndexType>::CooSparsityPattern(const CooSparsityPattern& sp)
    : dimensions_(sp.dimensions_), rowOffs_(sp.rowOffs_), colIdxs_(sp.colIdxs_),
      rowOffsV_(rowOffs_.view()), colIdxsV_(colIdxs_.view())
{}

#define NN_DECLARE_SPARSITY(TYPENAME) template class CooSparsityPattern<TYPENAME>

NN_FOR_ALL_INTEGER_TYPES(NN_DECLARE_SPARSITY);

} // namespace NeoN::la
