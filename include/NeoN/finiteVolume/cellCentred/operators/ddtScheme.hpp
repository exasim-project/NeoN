// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

namespace NeoN::finiteVolume::cellCentred
{

enum class DdtScheme
{
    None,
    SteadyState,
    BDF1,
    BDF2
};

} // namespace NeoN::finiteVolume::cellCentred
