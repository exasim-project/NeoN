// SPDX-FileCopyrightText: 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/input.hpp"
#include "NeoN/core/runtimeSelectionFactory.hpp"
#include "NeoN/linearAlgebra/linearSystem.hpp"

namespace NeoN::la
{


/* @brief A helper to collect statistics of the solver */
struct SolverStats
{
    int numIter;

    scalar initResNorm;

    scalar finalResNorm;

    void print(std::string solverName) const
    {
        std::cout << "Solver: " << solverName << " , Initial residual = " << initResNorm
                  << " , Final residual = " << finalResNorm
                  << " , No Iterations = " << numIter << std::endl;
    }
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

    virtual SolverStats solve(const LinearSystem<scalar, localIdx>&, Vector<scalar>&) const = 0;

    // virtual void
    // solve(const LinearSystem<ValueType, int>&, Vector<Vec3>& ) const = 0;

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

    SolverStats solve(const LinearSystem<scalar, localIdx>& ls, Vector<scalar>& field) const
    {
        SolverStats stats = solverInstance_->solve(ls, field);

        stats.print(solverInstance_->name());
        return stats;
    }

private:

    const Executor exec_;
    std::unique_ptr<SolverFactory> solverInstance_;
};

}
