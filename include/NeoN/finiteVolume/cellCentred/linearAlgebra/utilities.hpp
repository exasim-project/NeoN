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

template<typename ValueType, typename IndexType, typename SparsityType>
la::LinearSystem<ValueType, IndexType> assembleLinearSystem(
    dsl::Expression<ValueType> expr, VolumeField<ValueType>& rhs, const Dictionary& fvSchemes
)
{

    auto sp = SparsityPattern(SparsityPattern::readOrCreate(rhs.mesh()));

    la::LinearSystem<ValueType, IndexType> ls =
        la::createEmptyLinearSystem<ValueType, localIdx, SparsityPattern>(sp);

    // auto vol = rhs.mesh().cellVolumes().view();

    // Vector<ValueType> source(exec_, nCells);

    // expr.explicitOperation(source);
    // expr.explicitOperation(source, t, dt);

    // fill(ls.rhs(), zero<ValueType>());
    // fill(ls.matrix().values(), zero<ValueType>());

    // expr.implicitOperation(ls_);
    // // TODO rename implicitOperation -> assembleLinearSystem
    // expr.implicitOperation(ls_, t, dt);

    // auto rhsV = ls.rhs().view();
    // auto sourceV = source.view();

    // // we subtract the explicit source term from the rhs
    // NeoN::parallelFor(
    //     exec(),
    //     {0, rhs.size()},
    //     KOKKOS_LAMBDA(const localIdx i) { rhs[i] -= expSourceView[i] * vol[i]; }
    // );

    return ls;
}
}
