// SPDX-FileCopyrightText: 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp> // IWYU pragma: keep

namespace NeoN
{

template<typename T>
KOKKOS_INLINE_FUNCTION T one();

template<typename T>
KOKKOS_INLINE_FUNCTION T zero();


}
