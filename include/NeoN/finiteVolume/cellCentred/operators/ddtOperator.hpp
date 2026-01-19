// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/vector/vector.hpp"
#include "NeoN/core/executor/executor.hpp"
#include "NeoN/core/input.hpp"
#include "NeoN/dsl/operator.hpp"
#include "NeoN/linearAlgebra/linearSystem.hpp"
#include "NeoN/linearAlgebra/sparsityPattern.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"

namespace NeoN::finiteVolume::cellCentred
{

enum class DdtScheme
{
    None,
    BDF1,
    BDF2
};

template<typename ValueType>
class DdtOperator : public dsl::OperatorMixin<VolumeField<ValueType>>
{

public:

    using VectorValueType = ValueType;

    DdtOperator(dsl::Operator::Type termType, VolumeField<ValueType>& field);

    ~DdtOperator();

    void explicitOperation(Vector<ValueType>& source, scalar t, scalar dt) const;

    void implicitOperation(
        la::LinearSystem<ValueType, la::CSRMatrix<ValueType, localIdx>>& ls,
        const la::MatrixIterator<localIdx>& matrixIterator,
        scalar,
        scalar dt
    ) const;

    void bdf1Kernel(
        la::LinearSystem<ValueType, la::CSRMatrix<ValueType, localIdx>>& ls,
        const la::MatrixIterator<localIdx>& matrixIterator,
        scalar t,
        scalar dt
    ) const;

    void bdf2Kernel(
        la::LinearSystem<ValueType, la::CSRMatrix<ValueType, localIdx>>& ls,
        const la::MatrixIterator<localIdx>& matrixIterator,
        scalar t,
        scalar dt
    ) const;

    DdtScheme scheme() const noexcept { return scheme_; }

    void read(const Input&);

    std::string getName() const { return "DdtOperator"; }

private:

    // NOTE ddtOperator does not have a FactoryClass

    DdtScheme scheme_ {DdtScheme::BDF1};
};


} // namespace NeoN
