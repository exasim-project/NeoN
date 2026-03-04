// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#ifdef NF_WITH_MPI_SUPPORT
#if NF_WITH_GINKGO

#include <ginkgo/ginkgo.hpp>

#include "NeoN/core/dictionary.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/linearAlgebra/linearSystem.hpp"
#include "NeoN/linearAlgebra/solver.hpp"
#include "NeoN/linearAlgebra/utilities.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::la
{

/**
 * @class DistributedGinkgoSolver
 * @brief Solves a distributed linear system using Ginkgo's distributed CG solver.
 *
 * Converts the local LinearSystem (with ghost-column off-diagonal entries) to
 * a Ginkgo distributed::Matrix using global indices, then solves with CG +
 * Schwarz(Jacobi) preconditioner.
 */
class DistributedGinkgoSolver
{
public:

    DistributedGinkgoSolver(
        const Executor& exec, const Dictionary& solverCfg, const UnstructuredMesh& mesh
    );

    SolverStats solve(const LinearSystem<scalar>& ls, Vector<scalar>& x) const;

private:

    std::shared_ptr<gko::Executor> gkoExec_;
    const UnstructuredMesh& mesh_;
    Dictionary cfg_;
};

} // namespace NeoN::la

#endif // NF_WITH_GINKGO
#endif // NF_WITH_MPI_SUPPORT
