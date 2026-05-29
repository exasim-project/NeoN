// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/input.hpp"
#include "NeoN/core/runtimeSelectionFactory.hpp"
#include "NeoN/linearAlgebra/linearSystem.hpp"

namespace NeoN::la
{

struct SolverStatsEntry
{
    int numIter;
    scalar initResNorm;
    scalar finalResNorm;
    scalar solveTime;
};

/* @brief A helper to collect statistics of the solver */
struct SolverStats
{
    std::vector<SolverStatsEntry> entries;

    SolverStats() : entries() {}

    SolverStats(int numIter, scalar initResNorm, scalar finalResNorm, scalar solveTime)
        : entries({SolverStatsEntry {numIter, initResNorm, finalResNorm, solveTime}})
    {}

    SolverStats(SolverStatsEntry entry) : entries({entry}) {}
};

/* @class SolverFactory
**
*/
class SolverFactory :
    public RuntimeSelectionFactory<SolverFactory, Parameters<const Executor&, const Dictionary&>>
{
public:

    static std::unique_ptr<SolverFactory> create(const Executor& exec, const Dictionary& dict)
    {
        auto key = dict.get<std::string>("solver");
        SolverFactory::keyExistsOrError(key);
        return SolverFactory::table().at(key)(exec, dict);
    }

    static std::string name() { return "SolverFactory"; }

    SolverFactory(const Executor& exec) : exec_(exec) {};

    virtual SolverStats
    solve(const LinearSystem<scalar, CSRMatrix<scalar, localIdx>>&, Vector<scalar>&) const = 0;

    virtual SolverStats
    solve(const LinearSystem<Vec3, CSRMatrix<Vec3, localIdx>>&, Vector<Vec3>&) const = 0;

    virtual SolverStats
    solve(const LinearSystem<scalar, CSRMatrix<scalar, localIdx>, COOMatrix<scalar, localIdx>, Vec3>&, Vector<Vec3>&)
        const
    {
        NF_THROW("solve(scalar matrix, Vec3 rhs) not implemented for this solver");
    }

#ifdef NF_WITH_MPI_SUPPORT
    virtual SolverStats
    solveDist(const LinearSystem<scalar, CSRMatrix<scalar, localIdx>>&, Vector<scalar>&) const
    {
        NF_THROW("solveDist not implemented for this solver");
    }

    virtual SolverStats
    solveDist(const LinearSystem<Vec3, CSRMatrix<Vec3, localIdx>>&, Vector<Vec3>&) const
    {
        NF_THROW("solveDist not implemented for this solver");
    }
#endif

    // Pure virtual function for cloning
    virtual std::unique_ptr<SolverFactory> clone() const = 0;

protected:

    const Executor exec_;
};

class Solver
{

public:

    Solver(const Solver& solver)
        : exec_(solver.exec_), solverInstance_(solver.solverInstance_->clone()) {};

    Solver(Solver&& solver)
        : exec_(solver.exec_), solverInstance_(std::move(solver.solverInstance_)) {};

    Solver(const Executor& exec, std::unique_ptr<SolverFactory> solverInstance)
        : exec_(exec), solverInstance_(std::move(solverInstance)) {};

    Solver(const Executor& exec, const Dictionary& dict)
        : exec_(exec), solverInstance_(SolverFactory::create(exec, dict)) {};

    SolverStats
    solve(const LinearSystem<scalar, CSRMatrix<scalar, localIdx>>& ls, Vector<scalar>& field) const
    {
#ifdef NF_WITH_MPI_SUPPORT
        // Dispatch based on whether the job runs distributed (sizeRank > 1), NOT on
        // whether this rank has non-empty sendCounts: a rank without processor faces
        // must still join the distributed collectives or the job deadlocks.
        if (ls.commPattern().isDistributed()) return solverInstance_->solveDist(ls, field);
#endif
        return solverInstance_->solve(ls, field);
    }

    SolverStats
    solve(const LinearSystem<Vec3, CSRMatrix<Vec3, localIdx>>& ls, Vector<Vec3>& field) const
    {
#ifdef NF_WITH_MPI_SUPPORT
        if (ls.commPattern().isDistributed()) return solverInstance_->solveDist(ls, field);
#endif
        return solverInstance_->solve(ls, field);
    }

    /// @brief Solve a system with a scalar matrix and Vec3 right-hand side.
    /// @note This overload is **local-only**: it does not dispatch to solveDist and must not be
    ///       called on a distributed LinearSystem (one whose commPattern reports isDistributed()).
    ///       If distributed support is needed, add a solveDist virtual to SolverFactory and
    ///       implement it in every backend.
    SolverStats solve(
        const LinearSystem<scalar, CSRMatrix<scalar, localIdx>, COOMatrix<scalar, localIdx>, Vec3>&
            ls,
        Vector<Vec3>& field
    ) const
    {
#ifdef NF_WITH_MPI_SUPPORT
        NF_ASSERT(
            !ls.commPattern().isDistributed(),
            "solve(scalar matrix, Vec3 rhs) does not support distributed systems. "
            "Add a solveDist overload to SolverFactory to enable this."
        );
#endif
        return solverInstance_->solve(ls, field);
    }

private:

    const Executor exec_;
    std::unique_ptr<SolverFactory> solverInstance_;
};

}
