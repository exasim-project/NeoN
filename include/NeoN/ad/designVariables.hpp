// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/primitives/dual.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace NeoN::ad
{

/**
 * @class DesignVariables
 * @brief Registry mapping named design variables onto derivative slots.
 *
 * The registry mediates between two views of the same data:
 *   - the user, who wants named, typed handles (`nu`, `sourceStrength`);
 *   - the optimiser, which wants a flat, scaled vector of length nAlpha.
 *
 * Declaration returns an *active* Dual seeded in its own slot. From that point
 * the value is usable anywhere the underlying arithmetic type is usable, so no
 * AD-specific syntax appears in the equation the user writes.
 *
 * Scope of this MWE: scalar and small-array design variables (model
 * coefficients, operating conditions). Boundary- and volume-field design
 * variables need n_alpha to exceed what forward mode can carry and are
 * therefore deferred to the adjoint work; the declaration API is sketched at
 * the bottom of this header so the intended shape is on record.
 *
 * Mesh coordinates are explicitly *out* of scope: geometric quantities remain
 * plain NeoN::scalar so that any attempt to differentiate through mesh metrics
 * is a compile error rather than a silently wrong sensitivity.
 *
 * @tparam ValueType  underlying arithmetic type, typically NeoN::scalar
 * @tparam NDeriv     compile-time capacity, i.e. max number of design variables
 */
template<typename ValueType, int NDeriv>
class DesignVariables
{
public:

    using DualType = Dual<ValueType, NDeriv>;

    static constexpr int capacity = NDeriv;

    /**
     * @brief Declare a scalar design variable.
     * @param name   identifier, used for reporting and dictionary lookup
     * @param value  initial value
     * @param scale  optional characteristic magnitude used to normalise the
     *               flat gradient handed to the optimiser. Defaults to |value|,
     *               falling back to 1 for a zero initial value. Without this a
     *               gradient vector mixing nu ~ 1e-5 with Cs ~ 0.2 is badly
     *               conditioned.
     */
    DualType declare(const std::string& name, ValueType value, ValueType scale = ValueType(0))
    {
        if (static_cast<int>(names_.size()) >= NDeriv)
        {
            throw std::runtime_error(
                "DesignVariables: capacity " + std::to_string(NDeriv)
                + " exceeded when declaring '" + name + "'. Increase NDeriv, reduce the number of "
                  "design variables, or switch to reverse mode."
            );
        }
        const int slot = static_cast<int>(names_.size());
        names_.push_back(name);
        values_.push_back(value);

        ValueType s = scale;
        if (s == ValueType(0))
        {
            const ValueType a = (value < ValueType(0)) ? -value : value;
            s = (a > ValueType(0)) ? a : ValueType(1);
        }
        scales_.push_back(s);

        return DualType(value, slot);
    }

    /**
     * @brief Declare a small array of design variables sharing a base name.
     *
     * Used for parameterisations: the returned coefficients are combined by
     * ordinary expressions (e.g. a Bezier profile), so the chain rule through
     * the parameterisation is handled by the same machinery as the PDE, with
     * no parameterisation-specific AD code.
     */
    std::vector<DualType> declareArray(const std::string& name, const std::vector<ValueType>& values)
    {
        std::vector<DualType> out;
        out.reserve(values.size());
        for (std::size_t i = 0; i < values.size(); ++i)
        {
            out.push_back(declare(name + "[" + std::to_string(i) + "]", values[i]));
        }
        return out;
    }

    /** @brief Number of declared design variables, i.e. n_alpha. */
    int size() const { return static_cast<int>(names_.size()); }

    const std::string& name(int slot) const { return names_.at(static_cast<std::size_t>(slot)); }

    ValueType value(int slot) const { return values_.at(static_cast<std::size_t>(slot)); }

    ValueType scale(int slot) const { return scales_.at(static_cast<std::size_t>(slot)); }

    /**
     * @brief Extract the raw sensitivity dJ/dalpha_i from an evaluated functional.
     */
    ValueType gradient(const DualType& functional, int slot) const
    {
        checkSlot(slot);
        return functional.deriv(slot);
    }

    /**
     * @brief Extract the scaled gradient vector for an optimiser.
     *
     * Returns dJ/d(alpha_i / s_i) = s_i * dJ/dalpha_i, so the vector the
     * optimiser sees is O(1) regardless of the physical units of each variable.
     */
    std::vector<ValueType> scaledGradient(const DualType& functional) const
    {
        std::vector<ValueType> g;
        g.reserve(names_.size());
        for (int i = 0; i < size(); ++i)
        {
            g.push_back(scales_[static_cast<std::size_t>(i)] * functional.deriv(i));
        }
        return g;
    }

    /**
     * @brief Report design variables with exactly zero sensitivity.
     *
     * A declared variable with no path to the functional yields an identically
     * zero gradient. This is almost always a user error - a coefficient copied
     * into a passive ValueType somewhere along the chain, or a typo in the case
     * setup - and returning zeros silently is the worst possible outcome, since
     * the optimiser will simply never move that variable. Callers should treat
     * a non-empty result as a warning.
     */
    std::vector<std::string> passiveVariables(const DualType& functional) const
    {
        std::vector<std::string> dead;
        for (int i = 0; i < size(); ++i)
        {
            if (functional.deriv(i) == ValueType(0))
            {
                dead.push_back(names_[static_cast<std::size_t>(i)]);
            }
        }
        return dead;
    }

private:

    void checkSlot(int slot) const
    {
        if (slot < 0 || slot >= size())
        {
            throw std::out_of_range("DesignVariables: slot out of range");
        }
    }

    std::vector<std::string> names_;
    std::vector<ValueType> values_;
    std::vector<ValueType> scales_;
};

// ---------------------------------------------------------------------------
// Deferred API (adjoint work, not implemented here)
// ---------------------------------------------------------------------------
//
//   auto uIn = dv.boundaryField<Vector>("inlet", mesh.boundary("inlet"));
//   auto src = dv.volumeField<Vector>("forcing", mesh.zone("actuator"));
//
// These raise n_alpha to O(n_face) or O(n_cell), at which point forward mode
// costs n_alpha primal solves and reverse mode is the only viable choice.
// The registry contract above is deliberately mode-agnostic: gradient() and
// scaledGradient() keep the same signature once the sensitivity is produced by
// an adjoint sweep instead of read off a Dual.

} // namespace NeoN::ad
