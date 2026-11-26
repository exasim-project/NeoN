// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

/* @brief marks a type whose RTTI must be shared across shared-object boundaries.
 *
 * The `_neon` Python extension is built `-fvisibility=hidden` (nanobind's default), which by
 * default also hides the typeinfo emitted for a non-polymorphic class. `std::any` dispatches on
 * `type_info` identity, so a payload boxed inside `_neon` (e.g. a Dictionary built by the
 * Dictionary bindings) is unrecognisable to an `any_cast` compiled into libNeoN — `isDict()`
 * returns false and `get<TokenList>()` throws `bad any cast`. Forcing default visibility makes
 * the typeinfo a coalesced weak symbol so both objects agree on the type.
 *
 * Apply to any type that travels through `std::any` (or is caught as an exception) across the
 * libNeoN / `_neon` boundary.
 */
#if defined(_WIN32) || defined(__CYGWIN__)
#define NEON_TYPE_VISIBLE
#else
#define NEON_TYPE_VISIBLE __attribute__((visibility("default")))
#endif

/* @brief exports a function of libNeoN with default visibility.
 *
 * For entry points that must exist exactly once per process however a consumer is compiled
 * (`-fvisibility=hidden`, `-fvisibility-inlines-hidden`), such as the shared runtime-selection
 * registry. NeoN is a static library on Windows, so no dllexport is needed there.
 */
#if defined(_WIN32) || defined(__CYGWIN__)
#define NEON_EXPORT
#else
#define NEON_EXPORT __attribute__((visibility("default")))
#endif
