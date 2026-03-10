// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/copyTo.hpp"
#include "NeoN/linearAlgebra/matrix.hpp"

namespace NeoN::la
{

/**
 * @class DistributedMatrix
 * @brief Distributed matrix class
 */
template<typename ValueType, typename IndexType>
class DistributedMatrix : public SupportsCopyTo<DistributedMatrix<ValueType, IndexType>>
{
    using innerMtxType = CSRMatrix<ValueType, IndexType>;

    mpi::Environment env_;
    std::shared_ptr<innerMtxType> local_;
    std::shared_ptr<innerMtxType> nonLocal_;

public:

    /**
     * @brief Constructor for Matrix.
     *
     * @param locValues The non-zero values of the matrix.
     * @param locColIdxs The column indices for each non-zero value.
     * @param locRowOffs The starting index in values/colIdxs for each row.
     * @param nonLocValues The non-zero values of the matrix.
     * @param nonLocColIdxs The column indices for each non-zero value.
     * @param nonLocRowOffs The starting index in values/colIdxs for each row.
     */
    DistributedMatrix(
        const Vector<ValueType>&& locValues,
        const Vector<IndexType>&& locColIdxs,
        const Vector<IndexType>&& locRowOffs,
        const Vector<ValueType>&& nonLocValues,
        const Vector<IndexType>&& nonLocColIdxs,
        const Vector<IndexType>&& nonLocRowOffs,
        const mpi::Environment env
    )
        : local_(std::make_shared<innerMtxType>(
            std::move(locValues), std::move(locColIdxs), std::move(locRowOffs)
        )),
          nonLocal_(std::make_shared<innerMtxType>(
              std::move(nonLocValues), std::move(nonLocColIdxs), std::move(nonLocRowOffs)
          )),
          env_(env)
    {
        // FIXME  assert that nonLoc is empty if env is not initialized
    }

    /**
     * @brief Constructor for Matrix.
     *
     * @param localMatrix
     * @param nonLocalMatrix
     */
    DistributedMatrix(
        std::shared_ptr<innerMtxType> localMatrix,
        std::shared_ptr<innerMtxType> nonLocalMatrix,
        const mpi::Environment env
    )
        : local_(localMatrix), nonLocal_(nonLocalMatrix), env_(env)
    {
        // FIXME  assert that nonLoc is empty if env is not initialized
    }

    // getter

    [[nodiscard]] const Executor& exec() { return local_->exec(); }

    std::shared_ptr<innerMtxType> local() { return local_; }

    std::shared_ptr<const innerMtxType> local() const { return local_; }

    std::shared_ptr<innerMtxType> nonLocal() { return nonLocal_; }

    std::shared_ptr<const innerMtxType> nonLocal() const { return nonLocal_; }

    mpi::Environment environment() const { return env_; }

    [[nodiscard]] virtual DistributedMatrix copyToExecutor(Executor exec) const override
    {
        // FIXME implement
        return DistributedMatrix {
            std::make_shared<innerMtxType>(local_->copyToExecutor(exec)),
            std::make_shared<innerMtxType>(nonLocal_->copyToExecutor(exec)),
            env_
        };
    }
};


}
