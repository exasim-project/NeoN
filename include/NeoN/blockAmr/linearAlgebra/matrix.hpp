// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ginkgo/ginkgo.hpp>

#include <concepts>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

#include "NeoN/blockAmr/linearAlgebra/coefficients.hpp"
#include "NeoN/core/executor/executor.hpp"

namespace blockamr::la
{

/* @class Matrix
 * @brief Value-semantic holder for any type satisfying IsMatrix.
 *
 * The design is the type erasure NeoN already uses for dsl::SpatialOperator
 * (NeoN/dsl/spatialOperator.hpp): a private abstract `Concept` naming exactly the
 * public surface, a `Model<M>` holding one M BY VALUE, and copy through
 * `clone()`. Deliberately the same shape, not a new one -- a reader who knows
 * SpatialOperator knows this.
 *
 * The whole public surface FORWARDS; Matrix decides nothing. What a format
 * decides for itself is whether op() is assembled (CsrMatrix) or matrix-free
 * (MFFaceCoeffs), and nothing above this class needs to know which.
 *
 * Copying is a deep copy of the HELD FORMAT, which for the two formats in this
 * component means the copy shares their coefficient MultiFabs -- amrex::FabArray
 * has a deleted copy constructor, so a format cannot own its fields by value.
 * See faceCoeffMatrix.hpp.
 */
class Matrix
{
public:

    // The constrained converting constructor. The extra `requires` (which
    // SpatialOperator does not need, because no SpatialOperator is itself an
    // IsSpatialOperator) keeps `Matrix b {a};` on a NON-const Matrix lvalue from
    // preferring this template over the copy constructor and wrapping a Matrix
    // inside a Matrix -- Matrix satisfies IsMatrix, as the static_assert below
    // asserts, so without it that is what would happen.
    template<IsMatrix M>
        requires(!std::same_as<std::remove_cvref_t<M>, Matrix>)
    Matrix(M cls) : model_(std::make_unique<Model<M>>(std::move(cls)))
    {}

    Matrix(const Matrix& matrix) : model_(matrix.model_->clone()) {}

    Matrix(Matrix&& matrix) = default;

    Matrix& operator=(const Matrix& matrix)
    {
        model_ = matrix.model_->clone();
        return *this;
    }

    Matrix& operator=(Matrix&& matrix) = default;

    ~Matrix() = default;

    // The Ginkgo operator this matrix applies as. Assembled or matrix-free is
    // the format's business; a caller sees a LinOp either way.
    std::shared_ptr<const gko::LinOp> op() const { return model_->op(); }

    bool isAssembled() const { return model_->isAssembled(); }

    // Write handles onto the coefficients. NOT const: an assembled format has to
    // learn that its assembly is now stale, and this is where it does.
    MatrixCoefficients coefficients() { return model_->coefficients(); }

    void zero() { model_->zero(); }

    Symmetry symmetry() const { return model_->symmetry(); }

    // Rows THIS RANK owns, never the global row count.
    std::size_t localRows() const { return model_->localRows(); }

    const NeoN::Executor& executor() const { return model_->executor(); }

private:

    struct Concept
    {
        virtual ~Concept() = default;

        virtual std::shared_ptr<const gko::LinOp> op() const = 0;
        virtual bool isAssembled() const = 0;
        virtual MatrixCoefficients coefficients() = 0;
        virtual void zero() = 0;
        virtual Symmetry symmetry() const = 0;
        virtual std::size_t localRows() const = 0;
        virtual const NeoN::Executor& executor() const = 0;

        virtual std::unique_ptr<Concept> clone() const = 0;
    };

    template<IsMatrix M>
    struct Model : Concept
    {
        Model(M cls) : cls_(std::move(cls)) {}

        std::shared_ptr<const gko::LinOp> op() const override { return cls_.op(); }
        bool isAssembled() const override { return cls_.isAssembled(); }
        MatrixCoefficients coefficients() override { return cls_.coefficients(); }
        void zero() override { cls_.zero(); }
        Symmetry symmetry() const override { return cls_.symmetry(); }
        std::size_t localRows() const override { return cls_.localRows(); }
        const NeoN::Executor& executor() const override { return cls_.executor(); }

        std::unique_ptr<Concept> clone() const override { return std::make_unique<Model<M>>(cls_); }

        M cls_;
    };

    std::unique_ptr<Concept> model_;
};

// The erasure's forwarding surface IS the concept's surface: if a member is
// dropped, misspelled or given a different return type above, this fails here
// rather than at the first caller. (It also makes Matrix nestable in another
// Matrix, which the constrained constructor above deliberately prevents.)
static_assert(IsMatrix<Matrix>);

} // namespace blockamr::la
