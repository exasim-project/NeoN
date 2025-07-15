// SPDX-FileCopyrightText: 2024 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/core/parallelAlgorithms.hpp"
#include "NeoN/linearAlgebra/utilities.hpp"


namespace NeoN::la
{

void computeResidual(
    const CSRMatrix<scalar, localIdx>& mtx,
    const Vector<scalar>& bV,
    const Vector<scalar>& xV,
    Vector<scalar>& resV
)
{
    auto [res, b, x] = views(resV, bV, xV);
    const auto [coeffs, colIdxs, rowOffs] = mtx.view();

    NeoN::parallelFor(
        resV.exec(),
        {0, resV.size()},
        KOKKOS_LAMBDA(const localIdx rowi) {
            auto rowStart = rowOffs[rowi];
            auto rowEnd = rowOffs[rowi + 1];
            scalar sum = 0.0;
            for (localIdx coli = rowStart; coli < rowEnd; coli++)
            {
                sum += coeffs[coli] * x[colIdxs[coli]];
            }
            res[rowi] = sum - b[rowi];
        }
    );
}

}
