// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/linearAlgebra/blockVectorView.hpp"

namespace NeoN::la
{

/**
 * @class BlockVector
 * @brief A flat vector storing nBlocks contiguous blocks of nCells scalars each.
 *
 * Layout: [block0_cell0 ... block0_cellN, block1_cell0 ... block1_cellN, ...]
 * Block I occupies data[I * nCells ... (I+1) * nCells).
 */
class BlockVector
{

public:

    /**
     * @brief Construct a zero-initialized block vector.
     * @param exec Executor for memory allocation.
     * @param nBlocks Number of blocks (fields).
     * @param nCells Number of cells per block.
     */
    BlockVector(const Executor& exec, localIdx nBlocks, localIdx nCells);

    /**
     * @brief Construct a block vector filled with a uniform value.
     * @param exec Executor for memory allocation.
     * @param nBlocks Number of blocks (fields).
     * @param nCells Number of cells per block.
     * @param initVal Initial value for all elements.
     */
    BlockVector(const Executor& exec, localIdx nBlocks, localIdx nCells, scalar initVal);

    /**
     * @brief Access the flat (monolithic) vector.
     */
    [[nodiscard]] Vector<scalar>& vector();

    /** @brief Access the flat (monolithic) vector (const). */
    [[nodiscard]] const Vector<scalar>& vector() const;

    /** @brief Number of blocks. */
    [[nodiscard]] localIdx nBlocks() const;

    /** @brief Number of cells per block. */
    [[nodiscard]] localIdx nCells() const;

    /** @brief Total size (nBlocks * nCells). */
    [[nodiscard]] localIdx totalSize() const;

    /**
     * @brief Copy block i from the flat data into an external vector (scatter).
     * @param i Block index.
     * @param dst Destination vector (must have size >= nCells).
     */
    void copyBlockTo(localIdx i, Vector<scalar>& dst) const;

    /**
     * @brief Copy an external vector into block i of the flat data (gather).
     * @param i Block index.
     * @param src Source vector (must have size >= nCells).
     */
    void copyBlockFrom(localIdx i, const Vector<scalar>& src);

    /** @brief Device-safe view (only on lvalues). */
    [[nodiscard]] BlockVectorView view() &;

    /** @brief Prevent view() on temporaries. */
    BlockVectorView view() && = delete;

    /** @brief Get the executor. */
    [[nodiscard]] const Executor& exec() const;

private:

    Executor exec_;
    localIdx nBlocks_;
    localIdx nCells_;
    Vector<scalar> data_;
};

} // namespace NeoN::la
