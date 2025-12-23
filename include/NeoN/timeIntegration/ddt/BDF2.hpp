// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/timeIntegration/ddt/DdtScheme.hpp"

namespace NeoN::timeIntegration
{

/**
 * @brief BDF2 time-derivative scheme.
 */
class BDF2 final : public DdtScheme
{
public:

    int nSteps() const override { return 2; }

    scalar a0(scalar dt) const override { return scalar(1.5) / dt; }

    scalar a1(scalar dt) const override { return scalar(2.0) / dt; }

    scalar a2(scalar dt) const override { return scalar(-0.5) / dt; }

    // Startup = BDF1
    scalar a0Startup(scalar dt) const override { return scalar(1.0) / dt; }
    scalar a1Startup(scalar dt) const override { return scalar(1.0) / dt; }
};

} // namespace NeoN::timeIntegration::ddt
