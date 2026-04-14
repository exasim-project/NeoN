// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/macros.hpp"
#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/array.hpp"
#include "NeoN/linearAlgebra/csrSparsityPattern.hpp"

namespace NeoN::la
{

template<typename IndexType>
void CsrSparsityPattern<IndexType>::validate() const
{
    NF_ASSERT(rowOffs_.exec() == colIdxs_.exec(), "Executors are not the same");
    NF_ASSERT(rowOffs_.size() <= colIdxs_.size() + 1, "CSR size mismatch");

    // TODO add something like his test assert
    // auto rowCopy = rowOffs_.copyToHost();
    // NF_ASSERT(rowCopy.view()[rowOffs_.size()-1] == colIdxs_.size(), "broken rowOffs");
}

template<typename IndexType>
CsrSparsityPattern<IndexType>::CsrSparsityPattern(
    Vector<IndexType>&& colIdx, Vector<IndexType>&& rowOffs, Dimensions dim
)
    : dimensions_(dim), rowOffs_(std::move(rowOffs)), colIdxs_(std::move(colIdx)),
      rowOffsV_(rowOffs_.view()), colIdxsV_(colIdxs_.view())
{
    validate();
}

template<typename IndexType>
CsrSparsityPattern<IndexType>::CsrSparsityPattern(const CsrSparsityPattern& sp)
    : dimensions_(sp.dimensions_), rowOffs_(sp.rowOffs_), colIdxs_(sp.colIdxs_),
      rowOffsV_(rowOffs_.view()), colIdxsV_(colIdxs_.view())
{}

template<typename IndexType>
[[nodiscard]] const Vector<IndexType> rowsToRowOfs(Vector<IndexType>& rows)
{
    auto rowsHost = rows.copyToHost();
    const auto rowsV = rowsHost.view();
    const auto nnz = rowsV.size();

    IndexType maxRow = 0;
    // FIXME this needs to be in a parallel for
    for (localIdx i = 0; i < nnz; i++)
        if (rowsV[i] > maxRow) maxRow = rowsV[i];

    const IndexType nRows = maxRow + 1;
    Vector<IndexType> rowOffs(SerialExecutor {}, nRows + 1, IndexType(0));
    auto rowOffsV = rowOffs.view();

    // FIXME this needs to be in a parallel for
    for (localIdx i = 0; i < nnz; i++)
        rowOffsV[rowsV[i] + 1]++;
    // FIXME this needs to be in a parallel for
    for (IndexType r = 0; r < nRows; r++)
        rowOffsV[r + 1] += rowOffsV[r];

    return rowOffs.copyToExecutor(rows.exec());
}

#define NN_DECLARE_SPARSITY(TYPENAME) template class CsrSparsityPattern<TYPENAME>;

NN_FOR_ALL_INTEGER_TYPES(NN_DECLARE_SPARSITY);

} // namespace NeoN::la
