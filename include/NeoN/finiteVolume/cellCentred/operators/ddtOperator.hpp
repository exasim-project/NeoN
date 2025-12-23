// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
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
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "NeoN/timeIntegration/ddt/DdtScheme.hpp"
#include "NeoN/timeIntegration/ddt/BDF1.hpp"

namespace NeoN::finiteVolume::cellCentred
{

template<typename ValueType>
class DdtOperator : public dsl::OperatorMixin<VolumeField<ValueType>>
{

public:

    using VectorValueType = ValueType;

    DdtOperator(dsl::Operator::Type termType, VolumeField<ValueType>& field);

    ~DdtOperator();

    void explicitOperation(Vector<ValueType>& source, scalar, scalar dt) const;

    void implicitOperation(la::LinearSystem<ValueType, localIdx>& ls, scalar, scalar dt) const;

    void read(const Input&);

    const la::SparsityPattern& getSparsityPattern() const { return sparsityPattern_; }

    std::string getName() const { return "DdtOperator"; }

    const timeIntegration::DdtScheme& scheme() const
    {
        NF_ASSERT(scheme_ != nullptr, "ddt scheme not configured");
        return *scheme_;
    }

private:

    // NOTE ddtOperator does not have a FactoryClass
    const la::SparsityPattern& sparsityPattern_;

    static timeIntegration::BDF1 DEFAULT_BDF1_SCHEME;
    const timeIntegration::DdtScheme* scheme_ {&DEFAULT_BDF1_SCHEME};

    mutable bool firstTimeStep_ {true};
};


} // namespace NeoN
