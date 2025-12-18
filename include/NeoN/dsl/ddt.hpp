// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/fields/field.hpp"
#include "NeoN/finiteVolume/cellCentred/operators/ddtOperator.hpp"

namespace NeoN::dsl
{

/* @brief factory function to create a ddt term
 *
 * This injects the finite-volume DdtOperator into the DSL expression.
 * The choice of scheme (BDF1/BDF2) is read from fvSchemes.ddtSchemes.
 * Whether the operator is treated explicitly or implicitly is decided
 * by the time integrator.
 */
template<typename FieldType>
auto ddt(FieldType& field)
{
    return NeoN::finiteVolume::cellCentred::DdtOperator<FieldType>(field);
}

} // namespace NeoN
