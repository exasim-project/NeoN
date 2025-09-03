// SPDX-FileCopyrightText: 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

void computeResidual(
    const CsrMatrix<scalar, localIdx>& mtx,
    const Vector<scalar> b,
    const Vector<scalar>& x,
    Vector<scalar>& res
)
{
    auto [resV, bV, xV] = views(res, b, x);
    const auto [coeffs, colIdxs, rowOffs] = mtx.view();

    NeoN::parallelFor(
        res.exec(),
        {0, result.size()},
        KOKKOS_LAMBDA(const localIdx rowi) {
            auto rowStart = rowOffs[rowi];
            auto rowEnd = rowOffs[rowi + 1];
            scalar sum = 0.0;
            for (localIdx coli = rowStart; coli < rowEnd; coli++)
            {
                sum += coeffs[coli] * x[colIdxs[coli]];
            }
            values[rowi] = sum - b[rowi];
        }
    );
}
