// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoN authors
// TODO: move to cellCenred dsl?

#pragma once

#include "NeoN/linearAlgebra/linearSystem.hpp"
#include "NeoN/linearAlgebra/solver.hpp"
#include "NeoN/dsl/expression.hpp"
#include "NeoN/dsl/solver.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/linearAlgebra/sparsityPattern.hpp"
#include "NeoN/finiteVolume/cellCentred/linearAlgebra/utilities.hpp"

namespace dsl = NeoN::dsl;

namespace NeoN::finiteVolume::cellCentred
{

/*@brief extends expression by giving access to assembled matrix
 * @note used in neoIcoFOAM directly instead of dsl::expression
 * TODO: implement flag if matrix is assembled or not -> if not assembled call assemble
 * for dependent operations like discrete momentum fields
 * needs storage for assembled matrix? and whether update is needed like for rAU and HbyA
 */
template<typename ValueType, typename IndexType = localIdx>
class Expression
{
public:

    Expression(
        dsl::Expression<ValueType> expr,
        VolumeField<ValueType>& psi,
        const Dictionary& fvSchemes,
        const Dictionary& fvSolution
    )
        : psi_(psi), expr_(expr), fvSchemes_(fvSchemes), fvSolution_(fvSolution),
          sparsityPattern_(SparsityPattern::readOrCreate(psi.mesh())),
          ls_(la::createEmptyLinearSystem<ValueType, localIdx, SparsityPattern>(
              *sparsityPattern_.get()
          ))
    {
        expr_.read(fvSchemes_);
        // assemble();
    };

    Expression(const Expression& ls)
        : psi_(ls.psi_), expr_(ls.expr_), fvSchemes_(ls.fvSchemes_), fvSolution_(ls.fvSolution_),
          ls_(ls.ls_), sparsityPattern_(ls.sparsityPattern_) {};

    ~Expression() = default;

    [[nodiscard]] la::LinearSystem<ValueType, IndexType>& linearSystem() { return ls_; }
    [[nodiscard]] SparsityPattern& sparsityPattern()
    {
        if (!sparsityPattern_)
        {
            NF_THROW(std::string("fvcc:LinearSystem:sparsityPattern: sparsityPattern is null"));
        }
        return *sparsityPattern_;
    }

    VolumeField<ValueType>& getVector() { return this->psi_; }

    const VolumeField<ValueType>& getVector() const { return this->psi_; }

    [[nodiscard]] const la::LinearSystem<ValueType, IndexType>& linearSystem() const { return ls_; }
    [[nodiscard]] const SparsityPattern& sparsityPattern() const
    {
        if (!sparsityPattern_)
        {
            NF_THROW("fvcc:LinearSystem:sparsityPattern: sparsityPattern is null");
        }
        return *sparsityPattern_;
    }

    const Executor& exec() const { return ls_.exec(); }


    void assemble(scalar t, scalar dt)
    {
        auto vol = psi_.mesh().cellVolumes().view();
        auto expSource = expr_.explicitOperation(psi_.mesh().nCells());
        expr_.explicitOperation(expSource, t, dt);
        auto expSourceView = expSource.view();
        fill(ls_.rhs(), zero<ValueType>());
        fill(ls_.matrix().values(), zero<ValueType>());
        expr_.implicitOperation(ls_);
        // TODO rename implicitOperation -> assembleLinearSystem
        expr_.implicitOperation(ls_, t, dt);
        auto rhs = ls_.rhs().view();
        // we subtract the explicit source term from the rhs
        NeoN::parallelFor(
            exec(),
            {0, rhs.size()},
            KOKKOS_LAMBDA(const localIdx i) { rhs[i] -= expSourceView[i] * vol[i]; }
        );
    }

    void assemble()
    {
        if (expr_.temporalOperators().size() == 0 && expr_.spatialOperators().size() == 0)
        {
            NF_ERROR_EXIT("No temporal or implicit terms to solve.");
        }

        if (expr_.temporalOperators().size() > 0)
        {
            // integrate equations in time
            // NeoN::timeIntegration::TimeIntegration<VolumeField<ValueType>> timeIntegrator(
            //     fvSchemes_.subDict("ddtSchemes"), fvSolution_
            // );
            // timeIntegrator.solve(expr_, psi_, t, dt);
        }
        else
        {
            // solve sparse matrix system
            auto vol = psi_.mesh().cellVolumes().view();
            auto expSource = expr_.explicitOperation(psi_.mesh().nCells());
            auto expSourceView = expSource.view();

            ls_ = expr_.implicitOperation();
            auto rhs = ls_.rhs().view();
            // we subtract the explicit source term from the rhs
            NeoN::parallelFor(
                exec(),
                {0, rhs.size()},
                KOKKOS_LAMBDA(const localIdx i) { rhs[i] -= expSourceView[i] * vol[i]; }
            );
        }
    }

    // TODO unify with dsl/solver.hpp
    void solve(scalar, scalar)
    {
        // dsl::solve(expr_, psi_, t, dt, fvSchemes_, fvSolution_);
        if (expr_.temporalOperators().size() == 0 && expr_.spatialOperators().size() == 0)
        {
            NF_ERROR_EXIT("No temporal or implicit terms to solve.");
        }
        if (expr_.temporalOperators().size() > 0)
        {
            NF_ERROR_EXIT("Not implemented");
            //     // integrate equations in time
            //     NeoN::timeIntegration::TimeIntegration<VolumeField<ValueType>> timeIntegrator(
            //         fvSchemes_.subDict("ddtSchemes"), fvSolution_
            //     );
            //     timeIntegrator.solve(expr_, psi_, t, dt);
        }
        else
        {
            auto exec = psi_.exec();
            auto solver = NeoN::la::Solver(exec, fvSolution_);
            solver.solve(ls_, psi_.internalVector());
            // NF_ERROR_EXIT("No linear solver is available, build with -DNeoN_WITH_GINKGO=ON");
        }
    }

    void setReference(const IndexType refCell, ValueType refValue)
    {
        // TODO currently assumes that matrix is already assembled
        const auto diagOffset = sparsityPattern_->diagOffset().view();
        const auto rowOffs = ls_.matrix().rowOffs().view();
        auto rhs = ls_.rhs().view();
        auto values = ls_.matrix().values().view();
        NeoN::parallelFor(
            ls_.exec(),
            {refCell, refCell + 1},
            KOKKOS_LAMBDA(const std::size_t refCelli) {
                auto diagIdx = rowOffs[refCelli] + diagOffset[refCelli];
                auto diagValue = values[diagIdx];
                rhs[refCelli] += diagValue * refValue;
                values[diagIdx] += diagValue;
            }
        );
    }

private:

    VolumeField<ValueType>& psi_;
    dsl::Expression<ValueType> expr_;
    const Dictionary& fvSchemes_;
    const Dictionary& fvSolution_;
    std::shared_ptr<SparsityPattern> sparsityPattern_;
    la::LinearSystem<ValueType, IndexType> ls_;
};

/* @brief given a linear system consisting of A, b and x the operator computes the residual Ax-b
 *
 * @param[in] expr, the corresponding expression
 * @param[in] x, initial guess vector x
 * @return the corresponding residual stored in a VolumeField
 */
template<typename ValueType, typename IndexType = localIdx>
VolumeField<ValueType>
operator&(const Expression<ValueType, IndexType> expr, const VolumeField<ValueType>& x)
{
    VolumeField<ValueType> res(x);

    auto ls = expr.linearSystem();
    computeResidual(ls.matrix(), ls.rhs(), x.internalVector(), res.internalVector());
    return res;
}

}
