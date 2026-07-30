// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ginkgo/ginkgo.hpp>

#include <concepts>
#include <cstddef>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

#include "NeoN/blockAmr/core/fieldLevel.hpp"
#include "NeoN/blockAmr/core/meshLevel.hpp"
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

    // The `requires` keeps a Matrix out of its own Model. IsMatrix wants the coefficient DATA
    // MEMBERS, which an erasure cannot have (see the static_assert below), so `Matrix b {a};`
    // on a non-const lvalue already prefers the copy ctor -- this states it rather than
    // relying on it.
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

    // The layout the coefficients live on. Read-only, so nothing the format derives goes stale.
    const MeshLevel& mesh() const { return model_->mesh(); }

    // The three coefficient handles a caller writes through. NOT const, and each marks what the
    // format derives from them stale -- the fields are public on the format, so acquiring the
    // handle is the last moment a write can be observed.
    CellFieldLevel alpha() { return model_->alpha(); }

    FaceFieldLevel upper() { return model_->upper(); }

    std::optional<FaceFieldLevel> lower() { return model_->lower(); }

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
        virtual const MeshLevel& mesh() const = 0;
        virtual CellFieldLevel alpha() = 0;
        virtual FaceFieldLevel upper() = 0;
        virtual std::optional<FaceFieldLevel> lower() = 0;
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
        const MeshLevel& mesh() const override { return cls_.mesh; }
        // markStale() here rather than in Matrix, so Matrix's surface stays pure forwarding.
        CellFieldLevel alpha() override
        {
            cls_.markStale();
            return cls_.alpha;
        }
        FaceFieldLevel upper() override
        {
            cls_.markStale();
            return cls_.upper;
        }
        std::optional<FaceFieldLevel> lower() override
        {
            cls_.markStale();
            return cls_.lower;
        }
        void zero() override { cls_.zero(); }
        Symmetry symmetry() const override { return cls_.symmetry(); }
        std::size_t localRows() const override { return cls_.localRows(); }
        const NeoN::Executor& executor() const override { return cls_.exec; }
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

// Do not remove: the erasure deliberately DEPARTS from IsMatrix, which requires the coefficient
// fields as data MEMBERS -- an erasure has none, so it forwards them as accessors instead. This
// fires if the concept is ever loosened to accept accessors, which would also let a Matrix nest
// in a Matrix (see the constructor's `requires`).
static_assert(!IsMatrix<Matrix>);

} // namespace blockamr::la
