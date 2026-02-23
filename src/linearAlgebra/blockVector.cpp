// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/linearAlgebra/blockVector.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"

namespace NeoN::la
{

BlockVector::BlockVector(const Executor& exec, localIdx nBlocks, localIdx nCells)
    : exec_(exec), nBlocks_(nBlocks), nCells_(nCells), data_(exec, nBlocks * nCells, 0.0)
{}

BlockVector::BlockVector(const Executor& exec, localIdx nBlocks, localIdx nCells, scalar initVal)
    : exec_(exec), nBlocks_(nBlocks), nCells_(nCells), data_(exec, nBlocks * nCells, initVal)
{}

Vector<scalar>& BlockVector::vector() { return data_; }

const Vector<scalar>& BlockVector::vector() const { return data_; }

localIdx BlockVector::nBlocks() const { return nBlocks_; }

localIdx BlockVector::nCells() const { return nCells_; }

localIdx BlockVector::totalSize() const { return nBlocks_ * nCells_; }

void BlockVector::copyBlockTo(localIdx i, Vector<scalar>& dst) const
{
    auto srcView = data_.view();
    auto dstView = dst.view();
    localIdx offset = i * nCells_;
    localIdx n = nCells_;
    parallelFor(
        exec_,
        {0, n},
        NEON_LAMBDA(const localIdx ci) { dstView[ci] = srcView[offset + ci]; },
        "BlockVector_copyBlockTo"
    );
}

void BlockVector::copyBlockFrom(localIdx i, const Vector<scalar>& src)
{
    auto srcView = src.view();
    auto dstView = data_.view();
    localIdx offset = i * nCells_;
    localIdx n = nCells_;
    parallelFor(
        exec_,
        {0, n},
        NEON_LAMBDA(const localIdx ci) { dstView[offset + ci] = srcView[ci]; },
        "BlockVector_copyBlockFrom"
    );
}

BlockVectorView BlockVector::view() & { return BlockVectorView {data_.view(), nBlocks_, nCells_}; }

const Executor& BlockVector::exec() const { return exec_; }

} // namespace NeoN::la
