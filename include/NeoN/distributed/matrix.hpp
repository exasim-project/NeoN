// SPDX-FileCopyrightText: 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/copyTo.hpp"
#include "NeoN/linearAlgebra/CSRMatrix.hpp"

namespace NeoN::la
{

/**
 * @class Matrix
 * @brief Distributed matrix class
 */
class Matrix : public SupportsCopyTo<Matrix>
{
    using innerMtxType = CSRMatrix<scalar, localIdx>;

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
    Matrix(
        const Vector<scalar>&& locValues,
        const Vector<localIdx>&& locColIdxs,
        const Vector<localIdx>&& locRowOffs,
        const Vector<scalar>&& nonLocValues,
        const Vector<localIdx>&& nonLocColIdxs,
        const Vector<localIdx>&& nonLocRowOffs,
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
    Matrix(
        std::shared_ptr<innerMtxType> localMatrix,
        std::shared_ptr<innerMtxType> nonLocalMatrix,
        const mpi::Environment env
    )
        : local_(localMatrix), nonLocal_(nonLocalMatrix), env_(env)
    {
        // FIXME  assert that nonLoc is empty if env is not initialized
    }

    // getter

    std::shared_ptr<innerMtxType> local() { return local_; }

    std::shared_ptr<const innerMtxType> local() const { return local_; }

    std::shared_ptr<innerMtxType> nonLocal() { return nonLocal_; }

    std::shared_ptr<const innerMtxType> nonLocal() const { return nonLocal_; }

    mpi::Environment environment() const { return env_; }

    [[nodiscard]] virtual Matrix copyToExecutor(Executor exec) const override
    {
        // FIXME implement
        return Matrix {
            std::make_shared<innerMtxType>(local_->copyToExecutor(exec)),
            std::make_shared<innerMtxType>(nonLocal_->copyToExecutor(exec)),
            env_
        };
    }
};


}
