// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/portability.hpp"

namespace NeoN
{

template<typename T>
NEON_INLINE_FUNCTION T one();

template<typename T>
NEON_INLINE_FUNCTION T zero();

template<typename T>
NEON_INLINE_FUNCTION T inv(T);

}
