// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/macros.hpp"
#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/array.hpp"
#include "NeoN/linearAlgebra/sparsityPattern.hpp"

namespace NeoN::la
{

template<typename IndexType>
void SparsityPattern<IndexType>::validate() const
{
    NF_ASSERT(rowOffs_.exec() == colIdxs_.exec(), "Executors are not the same");
    NF_ASSERT(rowOffs_.size() <= colIdxs_.size() + 1, "CSR size mismatch");

    // TODO add something like his test assert
    // auto rowCopy = rowOffs_.copyToHost();
    // NF_ASSERT(rowCopy.view()[rowOffs_.size()-1] == colIdxs_.size(), "broken rowOffs");
}

template<typename IndexType>
SparsityPattern<IndexType>::SparsityPattern(Vector<IndexType>&& colIdx, Vector<IndexType>&& rowOffs)
    : rowOffs_(std::move(rowOffs)), colIdxs_(std::move(colIdx)), rowOffsV_(rowOffs_.view()),
      colIdxsV_(colIdxs_.view())
{
    validate();
}

template<typename IndexType>
SparsityPattern<IndexType>::SparsityPattern(const SparsityPattern& sp)
    : rowOffs_(sp.rowOffs_), colIdxs_(sp.colIdxs_), rowOffsV_(rowOffs_.view()),
      colIdxsV_(colIdxs_.view())
{}


#define NN_DECLARE_SPARSITY(TYPENAME) template class SparsityPattern<TYPENAME>

NN_FOR_ALL_INTEGER_TYPES(NN_DECLARE_SPARSITY);

} // namespace NeoN::la
