// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/macros.hpp"
#include "NeoN/core/containerFreeFunctions.hpp"
#include "NeoN/core/array.hpp"
#include "NeoN/linearAlgebra/ellSparsityPattern.hpp"

namespace NeoN::la
{

template<typename IndexType>
void EllSparsityPattern<IndexType>::validate() const
{
    NF_ASSERT(
        colIdxs_.size() == stride_ * numStoredElementsPerRow_,
        "ELL colIdxs size does not match stride * numStoredElementsPerRow"
    );
    NF_ASSERT(stride_ >= dimensions_.rows, "ELL stride must be at least the number of rows");
    NF_ASSERT(logicalNnz_ <= colIdxs_.size(), "ELL logical nnz exceeds padded storage size");
}

template<typename IndexType>
EllSparsityPattern<IndexType>::EllSparsityPattern(
    Vector<IndexType>&& colIdx,
    Dimensions dim,
    localIdx numStoredElementsPerRow,
    localIdx stride,
    localIdx logicalNnz
)
    : dimensions_(dim), colIdxs_(std::move(colIdx)),
      numStoredElementsPerRow_(numStoredElementsPerRow), stride_(stride), logicalNnz_(logicalNnz)
{
    validate();
}

template<typename IndexType>
EllSparsityPattern<IndexType>::EllSparsityPattern(const EllSparsityPattern& sp)
    : dimensions_(sp.dimensions_), colIdxs_(sp.colIdxs_),
      numStoredElementsPerRow_(sp.numStoredElementsPerRow_), stride_(sp.stride_),
      logicalNnz_(sp.logicalNnz_)
{}

#define NN_DECLARE_SPARSITY(TYPENAME) template class EllSparsityPattern<TYPENAME>;

NN_FOR_ALL_INTEGER_TYPES(NN_DECLARE_SPARSITY);

} // namespace NeoN::la
