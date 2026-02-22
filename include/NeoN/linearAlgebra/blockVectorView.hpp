// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp>

#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/view.hpp"

namespace NeoN::la
{

/**
 * @struct BlockVectorView
 * @brief Device-safe view into a flat block vector.
 *
 * Provides mdspan-like access: operator()(I) returns a subview for block I.
 * All blocks have the same size (nCells), so offsets are trivially I * nCells.
 */
struct BlockVectorView
{
    View<scalar> data; ///< Flat data of size nBlocks * nCells
    localIdx nBlocks;
    localIdx nCells;

    /**
     * @brief Access block I as a subview.
     * @param i The block index.
     * @return View<scalar> subview for block i.
     */
    KOKKOS_INLINE_FUNCTION
    View<scalar> operator()(localIdx i) const { return data.subview(i * nCells, nCells); }

    /**
     * @brief Direct element access into the flat data.
     * @param globalIndex The global index into the flat array.
     * @return Reference to the scalar value.
     */
    KOKKOS_INLINE_FUNCTION
    scalar& operator[](localIdx globalIndex) const { return data[globalIndex]; }
};

} // namespace NeoN::la
