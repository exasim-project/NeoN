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
 * @brief Value-semantic holder for any type satisfying IsMatrix, in NeoN
 *        dsl::SpatialOperator's erasure shape. The whole surface FORWARDS. Copying
 *        deep-copies the format but SHARES its coefficient MultiFabs (FabArray has none).
 */
class Matrix
{
public:

    // The `requires` is load-bearing: Matrix itself satisfies IsMatrix, so without it
    // `Matrix b {a};` on a non-const lvalue would nest a Matrix in a Matrix.
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

    // Assembled or matrix-free is the format's business; a caller sees a LinOp.
    std::shared_ptr<const gko::LinOp> op() const { return model_->op(); }

    bool isAssembled() const { return model_->isAssembled(); }

    // NOT const: this is where an assembled format learns its assembly is stale.
    MatrixCoefficients coefficients() { return model_->coefficients(); }

    void zero() { model_->zero(); }

    Symmetry symmetry() const { return model_->symmetry(); }

    // Rows THIS RANK owns, never the global row count.
    std::size_t localRows() const { return model_->localRows(); }

    const NeoN::Executor& executor() const { return model_->executor(); }

    // Built by the FORMAT from its own coefficients; null when it declines.
    std::shared_ptr<const gko::LinOp> makePrecond(const SolverConfig& config) const
    {
        return model_->makePrecond(config);
    }

    // For the message a caller raises when makePrecond declines; not a dispatch key.
    const char* name() const { return model_->name(); }

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
        virtual std::shared_ptr<const gko::LinOp> makePrecond(const SolverConfig&) const = 0;
        virtual const char* name() const = 0;

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
        std::shared_ptr<const gko::LinOp> makePrecond(const SolverConfig& config) const override
        {
            return cls_.makePrecond(config);
        }
        const char* name() const override { return cls_.name(); }

        std::unique_ptr<Concept> clone() const override { return std::make_unique<Model<M>>(cls_); }

        M cls_;
    };

    std::unique_ptr<Concept> model_;
};

// Do not remove: catches a dropped or retyped forwarding member here, not at a caller.
static_assert(IsMatrix<Matrix>);

} // namespace blockamr::la
