// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/linearAlgebra/blockSolve.hpp"
#include "NeoN/linearAlgebra/linearSystem.hpp"
#include "NeoN/linearAlgebra/matrix.hpp"

namespace NeoN::la
{

namespace
{

CSRMatrix<scalar, localIdx> buildMonolithicCSR(const Executor& targetExec, const BlockMatrix& bm)
{
    auto hostSparsity = bm.sparsity().copyToHost();
    auto hostValues = bm.values().copyToHost();

    const localIdx nb = bm.nBlocks();
    const localIdx nCells = bm.nCells();
    const localIdx nnzBlock = bm.nnz();
    const localIdx monoRows = nb * nCells;
    const localIdx monoNnz = nb * nb * nnzBlock;

    auto colView = hostSparsity.colIdxs().view();
    auto rowView = hostSparsity.rowOffs().view();
    auto valView = hostValues.view();

    std::vector<localIdx> monoRowOffs(static_cast<size_t>(monoRows + 1));
    std::vector<localIdx> monoColIdxs(static_cast<size_t>(monoNnz));
    std::vector<scalar> monoVals(static_cast<size_t>(monoNnz));

    localIdx pos = 0;
    monoRowOffs[0] = 0;

    for (localIdx I = 0; I < nb; ++I)
    {
        for (localIdx localRow = 0; localRow < nCells; ++localRow)
        {
            for (localIdx J = 0; J < nb; ++J)
            {
                for (localIdx k = rowView[localRow]; k < rowView[localRow + 1]; ++k)
                {
                    monoColIdxs[pos] = J * nCells + colView[k];
                    monoVals[pos] = valView[k * nb * nb + I + J * nb];
                    ++pos;
                }
            }
            monoRowOffs[I * nCells + localRow + 1] = pos;
        }
    }

    Executor hostExec = SerialExecutor();
    auto sp = std::make_shared<SparsityPattern<localIdx>>(
        Vector<localIdx>(hostExec, std::move(monoColIdxs)),
        Vector<localIdx>(hostExec, std::move(monoRowOffs))
    );
    CSRMatrix<scalar, localIdx> hostMtx(Vector<scalar>(hostExec, std::move(monoVals)), sp);

    return hostMtx.copyToExecutor(targetExec);
}

} // anonymous namespace

SolverStats solve(
    const BlockMatrix& matrix,
    const BlockVector& rhs,
    BlockVector& solution,
    const Dictionary& solverDict
)
{
    const Executor& exec = matrix.exec();

    auto monoMatrix = buildMonolithicCSR(exec, matrix);

    // Empty boundary system (BCs already incorporated into matrix and RHS)
    auto emptySparsity = std::make_shared<SparsityPattern<localIdx>>(
        Vector<localIdx>(exec, std::vector<localIdx> {}),
        Vector<localIdx>(exec, std::vector<localIdx> {0})
    );
    CSRMatrix<scalar, localIdx> boundaryMtx(Vector<scalar>(exec, 0, 0.0), emptySparsity);
    Vector<scalar> boundaryRhs(exec, 0, 0.0);

    LinearSystem<scalar> ls(monoMatrix, rhs.vector(), boundaryMtx, boundaryRhs, nullptr);

    Solver solver(exec, solverDict);
    return solver.solve(ls, solution.vector());
}

} // namespace NeoN::la
