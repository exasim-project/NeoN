// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/primitives/scalar.hpp"

namespace NeoN
{
namespace la
{
class LinearSystem;
}

class Field;

namespace timeIntegration
{
namespace ddt
{

/**
 * @brief Abstract base class for time-derivative discretisation schemes.
 *
 * This class ONLY describes how many historical time levels are required.
 * No assembly logic is introduced in Phase 1.
 *
 * Examples:
 *  - Euler      : nSteps() = 1
 *  - BDF2       : nSteps() = 2
 */
class DdtScheme
{
public:
    virtual ~DdtScheme() = default;

    /// Number of stored time levels required
    virtual int nSteps() const = 0;

    /**
     * @brief Add implicit ddt contribution to the linear system.
     *
     * Must reproduce existing behaviour for backward Euler.
     */
    virtual void addImplicit
    (
        la::LinearSystem& ls,
        const Field& field,
        scalar dt
    ) const = 0;
};

} // namespace ddt
} // namespace timeIntegration
} // namespace NeoN

