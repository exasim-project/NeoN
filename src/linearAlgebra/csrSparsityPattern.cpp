// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <utility>

#include "NeoN/core/macros.hpp"
#include "NeoN/linearAlgebra/csrSparsityPattern.hpp"
#include "NeoN/core/error.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/linearAlgebra/sparsityView.hpp"

namespace NeoN::la
{

template<typename IndexType>
void CsrSparsityPattern<IndexType>::validate() const
{
    NF_ASSERT(rowOffs_.exec() == colIdxs_.exec(), "Executors are not the same");

    // TODO add something like his test assert
    // NF_ASSERT(rowOffs_.size() <= colIdxs_.size() + 1, "CSR size mismatch");
    // auto rowCopy = rowOffs_.copyToHost();
    // NF_ASSERT(rowCopy.view()[rowOffs_.size()-1] == colIdxs_.size(), "broken rowOffs");
}

template<typename IndexType>
CsrSparsityPattern<IndexType>::CsrSparsityPattern(
    Vector<IndexType>&& colIdx, Vector<IndexType>&& rowOffs, Dimensions dim
)
    : dimensions_(dim), rowOffs_(std::move(rowOffs)), colIdxs_(std::move(colIdx))
{
    validate();
}

template<typename IndexType>
CsrSparsityPattern<IndexType>::CsrSparsityPattern(const CsrSparsityPattern& sp)
    : dimensions_(sp.dimensions_), rowOffs_(sp.rowOffs_), colIdxs_(sp.colIdxs_)
{}

#define NN_DECLARE_SPARSITY(TYPENAME) template class CsrSparsityPattern<TYPENAME>;

NN_FOR_ALL_INTEGER_TYPES(NN_DECLARE_SPARSITY);

} // namespace NeoN::la
