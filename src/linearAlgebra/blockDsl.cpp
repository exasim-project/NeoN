// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/linearAlgebra/blockDsl.hpp"

namespace NeoN::bdsl
{

// -- BlockSourceTerm ----------------------------------------------------------

BlockSourceTerm::BlockSourceTerm(scalar coefficient, std::string fieldName)
    : coefficient_(coefficient), fieldName_(std::move(fieldName))
{}

std::string BlockSourceTerm::getFieldName() const { return fieldName_; }

std::string BlockSourceTerm::getName() const { return "BlockSourceTerm"; }

scalar BlockSourceTerm::coefficient() const { return coefficient_; }

void BlockSourceTerm::implicitOperation(
    la::BlockMatrixView bmView,
    la::SparsityView<localIdx> spView,
    localIdx eqI,
    localIdx colJ,
    localIdx nCells,
    const Executor& exec
) const
{
    scalar coeff = coefficient_;
    parallelFor(
        exec,
        {0, nCells},
        NEON_LAMBDA(const localIdx celli) {
            localIdx k = spView.entry(celli, celli);
            bmView(k)(eqI, colJ) += coeff;
        },
        "BlockSourceTerm_implicitOperation"
    );
}


} // namespace NeoN::bdsl
