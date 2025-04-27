// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoN authors

#pragma once

#include "NeoN/core/dictionary.hpp"
#include "NeoN/dsl/expression.hpp"
#include "NeoN/linearAlgebra/linearSystem.hpp"

#include "NeoN/finiteVolume/cellCentred/linearAlgebra/sparsityPattern.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
[[nodiscard]] la::LinearSystem<ValueType, localIdx> assembleLinearSystem(
    dsl::Expression<ValueType> expr,
    VolumeField<ValueType>& rhs,
    const Dictionary& fvSchemes,
    scalar t,
    scalar dt
)
{
    auto sp = SparsityPattern(rhs.mesh());
    auto exec = expr.exec();

    la::LinearSystem<ValueType, localIdx> ls =
        la::createEmptyLinearSystem<ValueType, localIdx, SparsityPattern>(sp);

    auto nCells = rhs.size();
    Vector<ValueType> source(exec, nCells);

    fill(ls.rhs(), zero<ValueType>());
    fill(ls.matrix().values(), zero<ValueType>());

    auto [csrHV, rhsHV] = ls.view();

    expr.explicitOperation(source);
    expr.explicitOperation(source, t, dt);
    expr.implicitOperation(ls);
    expr.implicitOperation(ls, t, dt);

    auto [rhsV, sourceV] = views(ls.rhs(), source);

    // subtract the explicit source term from the rhs
    auto vol = rhs.mesh().cellVolumes().view();
    NeoN::parallelFor(
        exec, {0, rhs.size()}, KOKKOS_LAMBDA(const localIdx i) { rhsV[i] -= sourceV[i] * vol[i]; }
    );

    return ls;
}
}
