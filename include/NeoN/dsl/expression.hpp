// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <vector>

#include "NeoN/core/error.hpp"
#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/fields/field.hpp"
#include "NeoN/linearAlgebra/cooSparsityPattern.hpp"
#include "NeoN/linearAlgebra/csrSparsityPattern.hpp"
#include "NeoN/linearAlgebra/faceToMatrixAddress.hpp"
#include "NeoN/linearAlgebra/linearSystem.hpp"
#include "NeoN/dsl/spatialOperator.hpp"
#include "NeoN/dsl/temporalOperator.hpp"
#ifdef NF_WITH_MPI_SUPPORT
#include "NeoN/core/mpi/environment.hpp"
#endif

#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"

namespace NeoN::dsl
{

template<typename VectorType, typename IndexType>
struct PostAssemblyBase
{
    virtual ~PostAssemblyBase() = default;
    virtual void
    operator()(la::LinearSystem<VectorType, VectorType, la::CSRMatrix<VectorType, IndexType>>&)
        const {};

    /** @brief Apply to the segregated scalar-matrix / VectorType-rhs form (a scalar coefficient
     *         matrix with a VectorType right-hand side). Default no-op; functors that support the
     *         segregated form override this. A distinct name (rather than an operator() overload)
     *         avoids colliding with the same-type signature when VectorType == scalar. */
    virtual void applyScalarMatrix(la::LinearSystem<
                                   scalar,
                                   VectorType,
                                   la::CSRMatrix<scalar, IndexType>,
                                   la::COOMatrix<scalar, IndexType>>&) const {};
};

/**
 * @class SetReference
 * @brief Post-assembly functor that pins one cell's value to a reference, removing the
 *        constant null space that arises when all boundaries have Neumann (zero-gradient)
 *        conditions on operators such as laplacian or div+laplacian.
 *
 * Modifies the assembled linear system in-place:
 *   A[refCell, refCell] += A[refCell, refCell]   (doubles the diagonal)
 *   rhs[refCell]        += A[refCell, refCell] * refValue
 *
 * For distributed systems only the rank that owns the reference cell (assumed to be
 * rank 0 for local cell index 0) applies the modification; all other ranks skip it.
 * For non-distributed systems every rank applies it independently (each holds a full copy).
 * @TODO allow to set a refPoint instead of a refCell, make the refCell a global cellID
 */
template<typename ValueType, typename IndexType = localIdx>
class SetReference : public PostAssemblyBase<ValueType, IndexType>
{
public:

    SetReference(localIdx refCell, ValueType refValue) : refCell_(refCell), refValue_(refValue) {}

    void operator()(la::LinearSystem<ValueType, ValueType, la::CSRMatrix<ValueType, IndexType>>& ls
    ) const override
    {
#ifdef NF_WITH_MPI_SUPPORT
        // For distributed systems, only the rank owning refCell applies the constraint.
        // For non-distributed systems (each rank holds a full copy), every rank applies it.
        if (!ls.commPattern().sendCounts.empty())
        {
            mpi::Environment mpiEnv;
            if (mpiEnv.isInitialized() && mpiEnv.rank() != 0) return;
        }
#endif
        auto lsView = ls.view();
        const auto ma = ls.faceToMatrixAddress()->view(ls.matrix().sparsity()->rowOffs().view());
        auto refVal = refValue_;
        auto refCell = refCell_;
        parallelFor(
            ls.exec(),
            {refCell, refCell + 1},
            NEON_LAMBDA(const localIdx celli) {
                auto dIdx = ma.diagIdx(celli);
                auto diagVal = lsView.matrix.values[dIdx];
                lsView.rhs[celli] += diagVal * refVal;
                lsView.matrix.values[dIdx] += diagVal;
            },
            "SetReference"
        );
    }

    /** @brief Segregated scalar-matrix / ValueType-rhs form. The scalar diagonal scales the
     *         ValueType reference value (scalar * Vec3 broadcasts), so the same pin applies to
     *         every component of the right-hand side. */
    void applyScalarMatrix(la::LinearSystem<
                           scalar,
                           ValueType,
                           la::CSRMatrix<scalar, IndexType>,
                           la::COOMatrix<scalar, IndexType>>& ls) const override
    {
#ifdef NF_WITH_MPI_SUPPORT
        // For distributed systems, only the rank owning refCell applies the constraint.
        // For non-distributed systems (each rank holds a full copy), every rank applies it.
        if (!ls.commPattern().sendCounts.empty())
        {
            mpi::Environment mpiEnv;
            if (mpiEnv.isInitialized() && mpiEnv.rank() != 0) return;
        }
#endif
        auto lsView = ls.view();
        const auto ma = ls.faceToMatrixAddress()->view(ls.matrix().sparsity()->rowOffs().view());
        auto refVal = refValue_;
        auto refCell = refCell_;
        parallelFor(
            ls.exec(),
            {refCell, refCell + 1},
            NEON_LAMBDA(const localIdx celli) {
                auto dIdx = ma.diagIdx(celli);
                auto diagVal = lsView.matrix.values[dIdx];
                lsView.rhs[celli] += diagVal * refVal;
                lsView.matrix.values[dIdx] += diagVal;
            },
            "SetReference"
        );
    }

private:

    localIdx refCell_;
    ValueType refValue_;
};


/**
 * @class FixedValueConstraints
 * @brief Post-assembly functor that HARD-pins a set of cells to prescribed values — the
 *        equivalent of OpenFOAM's fvMatrix::setValues, which omega/epsilon wall functions
 *        apply through manipulateMatrix(). Unlike SetReference (which only removes a constant
 *        null space by doubling a single diagonal), this forces each constrained cell's
 *        solution exactly to its target.
 *
 * For every constrained cell the assembled row is rewritten in place:
 *   A[c, j] = 0  for all j != c   (zero the off-diagonals of row c)
 *   rhs[c]  = A[c, c] * value[c]
 * so the row reduces to  A[c,c] * x_c = A[c,c] * value[c]  =>  x_c = value[c], regardless of
 * the (relaxed, BC-augmented) diagonal magnitude. The column entries A[j, c] in other rows are
 * left intact, so neighbour equations correctly see the pinned value (more conservative than
 * upstream's full decouple, and valid because omega is solved with an asymmetric solver).
 *
 * The functor sweeps all nCells; a cell is constrained iff mask[cell] != 0, with value[cell]
 * holding its target. Both views are sized nCells and owned by the caller (must outlive use).
 */
template<typename ValueType, typename IndexType = localIdx>
class FixedValueConstraints : public PostAssemblyBase<ValueType, IndexType>
{
public:

    FixedValueConstraints(View<const scalar> mask, View<const ValueType> values, localIdx nCells)
        : mask_(mask), values_(values), nCells_(nCells)
    {}

    void operator()(la::LinearSystem<ValueType, ValueType, la::CSRMatrix<ValueType, IndexType>>& ls
    ) const override
    {
        auto lsView = ls.view();
        const auto ma = ls.faceToMatrixAddress()->view(ls.matrix().sparsity()->rowOffs().view());
        auto mask = mask_;
        auto vals = values_;
        parallelFor(
            ls.exec(),
            {0, nCells_},
            NEON_LAMBDA(const localIdx celli) {
                if (mask[celli] == scalar(0)) return;
                const auto dIdx = ma.diagIdx(celli);
                const ValueType diagVal = lsView.matrix.values[dIdx];
                const auto rowStart = ma.rowOffs[celli];
                const auto rowEnd = ma.rowOffs[celli + 1];
                for (auto o = rowStart; o < rowEnd; ++o)
                {
                    if (o != dIdx) lsView.matrix.values[o] = zero<ValueType>();
                }
                lsView.rhs[celli] = diagVal * vals[celli];
            },
            "FixedValueConstraints"
        );
    }

private:

    View<const scalar> mask_;
    View<const ValueType> values_;
    localIdx nCells_;
};


template<typename ValueType, typename IndexType = localIdx>
class Expression
{
public:

    using ExpressionValueType = ValueType;

    Expression(const Executor& exec) : exec_(exec), temporalOperators_(), spatialOperators_() {}

    Expression(const Expression& exp)
        : exec_(exp.exec_), temporalOperators_(exp.temporalOperators_),
          spatialOperators_(exp.spatialOperators_)
    {}

    Expression(const SpatialOperator<ValueType>& oper)
        : exec_(oper.exec()), temporalOperators_(), spatialOperators_()
    {
        spatialOperators_.push_back(oper);
    }

    Expression& operator=(const Expression& exp)
    {
        if (this == &exp)
        {
            return *this;
        }
        NF_ASSERT(exec_ == exp.exec_, "Executors are not the same");
        temporalOperators_ = exp.temporalOperators_;
        spatialOperators_ = exp.spatialOperators_;
        return *this;
    }


    Expression(const TemporalOperator<ValueType>& oper)
        : exec_(oper.exec()), temporalOperators_(), spatialOperators_()
    {
        temporalOperators_.push_back(oper);
    }

    /* @brief dispatch read call to operator */
    void read(const Dictionary& input)
    {
        for (auto& op : temporalOperators_)
        {
            op.read(input);
        }
        for (auto& op : spatialOperators_)
        {
            op.read(input);
        }
    }

    /* @brief perform all explicit operation and accumulate the result */
    Vector<ValueType> explicitOperation(localIdx nCells) const
    {
        Vector<ValueType> source(exec_, nCells, zero<ValueType>());
        return explicitOperation(source);
    }

    /* @brief perform all explicit operation and accumulate the result */
    Vector<ValueType> explicitOperation(Vector<ValueType>& source) const
    {
        for (auto& op : spatialOperators_)
        {
            if (op.getType() == Operator::Type::Explicit)
            {
                op.explicitOperation(source);
            }
        }
        return source;
    }

    Vector<ValueType> explicitOperation(Vector<ValueType>& source, scalar t, scalar dt) const
    {
        for (auto& op : temporalOperators_)
        {
            if (op.getType() == Operator::Type::Explicit)
            {
                op.explicitOperation(source, t, dt);
            }
        }
        return source;
    }

    /** @brief compute matrix coefficients based on all spatial operators */
    template<typename AssemblyType = ValueType>
    void assembleSpatialOperator(la::LinearSystem<AssemblyType, ValueType>& ls) const
    {
        for (auto& op : spatialOperators_)
        {
            if (op.getType() == Operator::Type::Implicit)
            {
                op.implicitOperation(ls);
            }
        }
    }

    /** @brief compute matrix coefficients based on all temporal operators
     * assemble directly into linear system
     */
    template<typename AssemblyType = ValueType>
    void assembleTemporalOperator(
        la::LinearSystem<AssemblyType, ValueType>& ls, scalar t, scalar dt
    ) const
    {
        for (auto& op : temporalOperators_)
        {
            if (op.getType() == Operator::Type::Implicit)
            {
                op.implicitOperation(ls, t, dt);
            }
        }
    }

    /*@brief subtract explicit source terms from the linear system rhs, scaled by cell volumes */
    template<typename AssemblyType = ValueType>
    void assembleExplicitSource(
        la::LinearSystem<AssemblyType, ValueType>& ls, const UnstructuredMesh& mesh
    ) const
    {
        auto expTmp = explicitOperation(static_cast<localIdx>(mesh.nCells()));
        auto [vol, expSource, rhs] = views(mesh.cellVolumes(), expTmp, ls.rhs());
        parallelFor(
            ls.exec(),
            {0, static_cast<localIdx>(rhs.size())},
            NEON_LAMBDA(const localIdx i) { rhs[i] -= expSource[i] * vol[i]; }
        );
    }

    /** @brief construct a linear system and force assembly including explicit source terms
     *
     * @param ps post-assembly functors applied to the system after assembly
     * @return the assembled linear system
     */
    template<typename AssemblyType = ValueType>
    la::LinearSystem<AssemblyType, ValueType> assemble(
        const UnstructuredMesh& mesh,
        scalar t,
        scalar dt,
        std::vector<const PostAssemblyBase<ValueType, IndexType>*> ps = {}
    ) const
    {
        auto ls = la::createEmptyLinearSystem<AssemblyType, ValueType>(mesh);
        assemble<AssemblyType>(t, dt, ls, mesh, ps);
        return ls;
    }

    /** @brief assemble into a given linear system including explicit source terms
     *
     * @param ps post-assembly functors applied to the system after assembly
     */
    template<typename AssemblyType = ValueType>
    void assemble(
        scalar t,
        scalar dt,
        la::LinearSystem<AssemblyType, ValueType>& ls,
        const UnstructuredMesh& mesh,
        std::vector<const PostAssemblyBase<ValueType, IndexType>*> ps = {}
    ) const
    {
        assemble<AssemblyType>(t, dt, ls, ps);
        assembleExplicitSource(ls, mesh);
    }

    /* @brief assemble into a given linear system (implicit operators only, no explicit sources)
     *
     * @param ps post-assembly functors applied to the system after assembly
     */
    template<typename AssemblyType = ValueType>
    void assemble(
        scalar t,
        scalar dt,
        la::LinearSystem<AssemblyType, ValueType>& ls,
        std::vector<const PostAssemblyBase<ValueType, IndexType>*> ps = {}
    ) const
    {
        assembleSpatialOperator(ls);         // add spatial operator
        assembleTemporalOperator(ls, t, dt); // add temporal operators

        // Post-assembly functors apply on the same-type form via operator(); the segregated
        // scalar-matrix / ValueType-rhs form dispatches to applyScalarMatrix instead.
        if constexpr (std::is_same_v<AssemblyType, ValueType>)
        {
            for (const auto* p : ps)
            {
                (*p)(ls);
            }
        }
        else if constexpr (std::is_same_v<AssemblyType, scalar>)
        {
            for (const auto* p : ps)
            {
                p->applyScalarMatrix(ls);
            }
        }
    }

    void addOperator(const SpatialOperator<ValueType>& oper) { spatialOperators_.push_back(oper); }

    void addOperator(const TemporalOperator<ValueType>& oper)
    {
        temporalOperators_.push_back(oper);
    }

    void addExpression(const Expression& equation)
    {
        for (auto& op : equation.temporalOperators_)
        {
            temporalOperators_.push_back(op);
        }
        for (auto& op : equation.spatialOperators_)
        {
            spatialOperators_.push_back(op);
        }
    }

    /**@brief returns operator of given type and name exists */
    template<typename OperatorType, Operator::Type Type>
    bool hasOperatorOfType(const std::string& name) const
    {
        auto opType = Type;
        auto matchNameAndType = [name, opType](const auto& op)
        { return op.getName() == name && op.getType() == opType; };
        if constexpr (std::is_same_v<OperatorType, SpatialOperator<ValueType>>)
        {
            return std::ranges::any_of(spatialOperators_, matchNameAndType);
        }
        else if constexpr (std::is_same_v<OperatorType, TemporalOperator<ValueType>>)
        {
            return std::ranges::any_of(temporalOperators_, matchNameAndType);
        }
        return false;
    }

    /**@brief returns whether the expression contains an operator with a given name */
    template<Operator::Type Type>
    bool hasOperator(const std::string& name) const
    {
        return hasOperatorOfType<SpatialOperator<ValueType>, Type>(name)
            || hasOperatorOfType<TemporalOperator<ValueType>, Type>(name);
    }

    /**@brief returns operator of given type and name */
    template<typename OperatorType, Operator::Type Type>
    OperatorType& getOperator(const std::string& name)
    {
        if (!hasOperatorOfType<OperatorType, Type>(name))
        {
            throw std::runtime_error {"No operator with given name and type found"};
        }
        auto opType = Type;
        auto matchNameAndType = [name, opType](const auto& op)
        { return op.getName() == name && op.getType() == opType; };
        if constexpr (std::is_same_v<OperatorType, SpatialOperator<ValueType>>)
        {
            return *std::ranges::find_if(spatialOperators_, matchNameAndType);
        }
        else if constexpr (std::is_same_v<OperatorType, TemporalOperator<ValueType>>)
        {
            return *std::ranges::find_if(temporalOperators_, matchNameAndType);
        }
        throw std::runtime_error {"Unknown operator type"};
        // should never be reached, shut up compiler warning
        return spatialOperators_[0];
    }

    /**@brief removes operator of given name */
    template<Operator::Type Type>
    void dropOperator(const std::string& name)
    {
        if (!hasOperator<Type>(name))
        {
            throw std::runtime_error {"No operator with given name and type found"};
        }
        auto opType = Type;
        auto matchNameAndType = [name, opType](const auto& op)
        { return op.getName() == name && op.getType() == opType; };
        if (hasOperatorOfType<SpatialOperator<ValueType>, Type>(name))
        {
            std::erase_if(spatialOperators_, matchNameAndType);
        }
        else
        {
            std::erase_if(temporalOperators_, matchNameAndType);
        }
    }

    /* @brief getter for the total number of terms in the equation */
    localIdx size() const
    {
        return static_cast<localIdx>(temporalOperators_.size() + spatialOperators_.size());
    }

    // getters
    const std::vector<TemporalOperator<ValueType>>& temporalOperators() const
    {
        return temporalOperators_;
    }

    const std::vector<SpatialOperator<ValueType>>& spatialOperators() const
    {
        return spatialOperators_;
    }

    std::vector<TemporalOperator<ValueType>>& temporalOperators() { return temporalOperators_; }

    std::vector<SpatialOperator<ValueType>>& spatialOperators() { return spatialOperators_; }

    const Executor& exec() const { return exec_; }

private:

    const Executor exec_;

    std::vector<TemporalOperator<ValueType>> temporalOperators_;

    std::vector<SpatialOperator<ValueType>> spatialOperators_;
};

template<typename ValueType>
[[nodiscard]] inline Expression<ValueType>
operator+(Expression<ValueType> lhs, const Expression<ValueType>& rhs)
{
    lhs.addExpression(rhs);
    return lhs;
}

template<typename ValueType>
[[nodiscard]] inline Expression<ValueType>
operator+(Expression<ValueType> lhs, const SpatialOperator<ValueType>& rhs)
{
    lhs.addOperator(rhs);
    return lhs;
}

template<typename leftOperator, typename rightOperator>
[[nodiscard]] inline Expression<typename leftOperator::VectorValueType>
operator+(leftOperator lhs, rightOperator rhs)
{
    using ValueType = typename leftOperator::VectorValueType;
    Expression<ValueType> expr(lhs.exec());
    expr.addOperator(lhs);
    expr.addOperator(rhs);
    return expr;
}

template<typename ValueType>
[[nodiscard]] inline Expression<ValueType> operator*(scalar scale, const Expression<ValueType>& es)
{
    Expression<ValueType> expr(es.exec());
    for (const auto& oper : es.temporalOperators())
    {
        expr.addOperator(scale * oper);
    }
    for (const auto& oper : es.spatialOperators())
    {
        expr.addOperator(scale * oper);
    }
    return expr;
}


template<typename ValueType>
[[nodiscard]] inline Expression<ValueType>
operator-(Expression<ValueType> lhs, const Expression<ValueType>& rhs)
{
    lhs.addExpression(-1.0 * rhs);
    return lhs;
}

template<typename ValueType>
[[nodiscard]] inline Expression<ValueType>
operator-(Expression<ValueType> lhs, const SpatialOperator<ValueType>& rhs)
{
    lhs.addOperator(-1.0 * rhs);
    return lhs;
}

template<typename leftOperator, typename rightOperator>
[[nodiscard]] inline Expression<typename leftOperator::VectorValueType>
operator-(leftOperator lhs, rightOperator rhs)
{
    using ValueType = typename leftOperator::VectorValueType;
    Expression<ValueType> expr(lhs.exec());
    expr.addOperator(lhs);
    expr.addOperator(Coeff(-1) * rhs);
    return expr;
}


} // namespace dsl
