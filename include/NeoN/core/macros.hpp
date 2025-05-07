// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#pragma once

#include "NeoN/core/primitives/vec3.hpp"

/* @brief given a c macro the macro gets called for all regularly used integer types
 * this is used to instantiate templates for our container types
 */
#define NN_FOR_ALL_INTEGER_TYPES(_macro)                                                           \
    _macro(uint32_t);                                                                              \
    _macro(uint64_t);                                                                              \
    _macro(int32_t);                                                                               \
    _macro(int64_t)

/* @brief given a c macro the macro gets called for all scalar types
 * like float and double
 *
 */
#define NN_FOR_ALL_SCALAR_TYPES(_macro)                                                            \
    _macro(float);                                                                                 \
    _macro(double)

/* @brief given a c macro the macro gets called for all valid value types
 * which includes scalar and vec3 types
 *
 */
#define NN_FOR_ALL_VALUE_TYPES(_macro)                                                             \
    NN_FOR_ALL_SCALAR_TYPES(_macro);                                                               \
    _macro(Vec3)
