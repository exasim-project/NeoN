// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/primitives/scalar.hpp"

namespace NeoN
{
namespace timeIntegration
{

/**
 * @brief Time-derivative discretisation policy.
 *
 * Provides coefficients for implicit ddt assembly.
 * Does NOT perform any loops or Kokkos execution.
 */
class DdtScheme
{
public:

    virtual ~DdtScheme() = default;

    /// Number of historical time levels required
    virtual int nSteps() const = 0;

    /**
     * @brief Coefficient multiplying phi^{n+1}
     */
    virtual scalar a0(scalar dt) const = 0;

    /**
     * @brief Coefficient multiplying phi^{n}
     */
    virtual scalar a1(scalar dt) const = 0;

    /**
     * @brief Coefficient multiplying phi^{n-1}
     * default: unused
     */
    virtual scalar a2(scalar) const { return scalar(0); }

    /**
     * @brief Coefficients for schemes using multiple time levels
     * at the first timestep (startup phase) when the old time level phi^{n-1} is
     * not present yet
     */
    virtual scalar a0Startup(scalar dt) const { return a0(dt); }
    virtual scalar a1Startup(scalar dt) const { return a1(dt); }
};

} // namespace timeIntegration
} // namespace NeoN
