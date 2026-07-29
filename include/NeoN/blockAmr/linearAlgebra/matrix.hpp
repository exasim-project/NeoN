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
 * Same type-erasure shape as NeoN's dsl::SpatialOperator
 * (NeoN/dsl/spatialOperator.hpp), deliberately: private abstract `Concept`
 * naming exactly the public surface, `Model<M>` holding one M BY VALUE, copy
 * through `clone()`. The whole surface FORWARDS; Matrix decides nothing --
 * including whether op() is assembled (CsrMatrix) or matrix-free (MFFaceCoeffs).
 *
 * Copying deep-copies the HELD FORMAT, but the copy SHARES its coefficient
 * MultiFabs: amrex::FabArray has a deleted copy constructor, so a format cannot
 * own its fields by value. See faceCoeffMatrix.hpp.
 */
class Matrix
{
public:

    // The `requires` is load-bearing: Matrix itself satisfies IsMatrix (see the
    // static_assert below), so without it `Matrix b {a};` on a NON-const lvalue
    // would prefer this template over the copy ctor and nest a Matrix in a Matrix.
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

    // Write handles onto the coefficients. NOT const: this is where an assembled
    // format learns that its assembly is now stale.
    MatrixCoefficients coefficients() { return model_->coefficients(); }

    void zero() { model_->zero(); }

    Symmetry symmetry() const { return model_->symmetry(); }

    // Rows THIS RANK owns, never the global row count.
    std::size_t localRows() const { return model_->localRows(); }

    const NeoN::Executor& executor() const { return model_->executor(); }

    // Built by the FORMAT from its own coefficients; null when it declines
    // (coefficients.hpp).
    std::shared_ptr<const gko::LinOp> makePrecond(const SolverConfig& config) const
    {
        return model_->makePrecond(config);
    }

    // For the message a caller raises when makePrecond declines. Not a dispatch
    // key -- nothing branches on it.
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

// Do not remove: catches a dropped, misspelled or retyped forwarding member here
// rather than at the first caller. It is also why the ctor's `requires` is needed.
static_assert(IsMatrix<Matrix>);

} // namespace blockamr::la
