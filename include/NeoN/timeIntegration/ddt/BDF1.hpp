// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/timeIntegration/ddt/DdtScheme.hpp"

namespace NeoN
{
namespace timeIntegration
{

class BDF1 final : public DdtScheme
{
public:

    int nSteps() const override { return 1; }

    scalar a0(scalar dt) const override { return scalar(1.0) / dt; }

    scalar a1(scalar dt) const override { return scalar(1.0) / dt; }
};

} // namespace timeIntegration
} // namespace NeoN
