// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors
#pragma once

#include <vector>

#include "NeoN/core/primitives/label.hpp"
#include "NeoN/fields/fieldTypeDefs.hpp"

namespace NeoN
{

/**
 * @class BoundaryMesh
 * @brief Represents boundaries of an unstructured mesh.
 *
 * The BoundaryMesh class stores information about the boundary faces and their
 * properties in an unstructured mesh. It provides access to various fields such
 * as face cells, face centres, face normals, face areas normals, magnitudes of
 * face areas normals, delta vectors, weights, delta coefficients, and offsets.
 *
 * The class also provides getter methods to access the individual fields and
 * their components.
 *
 * @tparam Executor The type of the executor used for computations.
 */
class BoundaryMesh
{
public:

    /**
     * @brief Constructor for the BoundaryMesh class.
     *
     * @param exec The executor used for computations.
     * @param faceCells A field with the neighbouring cell of each boundary
     * face.
     * @param Cf A field of face centres.
     * @param Cn A field of neighbor cell centers.
     * @param Sf A field of face areas normals.
     * @param magSf A field of magnitudes of face areas normals.
     * @param nf A field of face unit normals.
     * @param delta A field of delta vectors.
     * @param weights A field of weights used in cell to face interpolation.
     * @param deltaCoeffs A field of cell to face distances.
     * @param offset The offset of the boundary faces.
     */
    BoundaryMesh(
        const Executor& exec,
        labelVector faceCells,
        vectorVector cf,
        vectorVector cn,
        vectorVector sf,
        scalarVector magSf,
        vectorVector nf,
        vectorVector delta,
        scalarVector weights,
        scalarVector deltaCoeffs,
        std::vector<localIdx> offset
    );


    /**
     * @brief Get the field of face cells.
     *
     * @return A constant reference to the field of face cells.
     */
    const labelVector& faceCells() const;

    // TODO either dont mix return types, ie dont use view and Vector
    // for functions with same name
    /**
     * @brief Get a view of face cells for a specific boundary face.
     *
     * @param i The index of the boundary face.
     * @return A view of face cells for the specified boundary face.
     */
    View<const label> faceCells(const localIdx i) const;

    /**
     * @brief Get the field of face centres.
     *
     * @return A constant reference to the field of face centres.
     */
    const vectorVector& cf() const;

    /**
     * @brief Get a view of face centres for a specific boundary face.
     *
     * @param i The index of the boundary face.
     * @return A view of face centres for the specified boundary face.
     */
    View<const Vec3> cf(const localIdx i) const;

    /**
     * @brief Get the field of face normals.
     *
     * @return A constant reference to the field of face normals.
     */
    const vectorVector& cn() const;

    /**
     * @brief Get a view of face normals for a specific boundary face.
     *
     * @param i The index of the boundary face.
     * @return A view of face normals for the specified boundary face.
     */
    View<const Vec3> cn(const localIdx i) const;

    /**
     * @brief Get the field of face areas normals.
     *
     * @return A constant reference to the field of face areas normals.
     */
    const vectorVector& sf() const;

    /**
     * @brief Get a view of face areas normals for a specific boundary face.
     *
     * @param i The index of the boundary face.
     * @return A view of face areas normals for the specified boundary face.
     */
    View<const Vec3> sf(const localIdx i) const;

    /**
     * @brief Get the field of magnitudes of face areas normals.
     *
     * @return A constant reference to the field of magnitudes of face areas
     * normals.
     */
    const scalarVector& magSf() const;

    /**
     * @brief Get a view of magnitudes of face areas normals for a specific
     * boundary face.
     *
     * @param i The index of the boundary face.
     * @return A view of magnitudes of face areas normals for the specified
     * boundary face.
     */
    View<const scalar> magSf(const localIdx i) const;

    /**
     * @brief Get the field of face unit normals.
     *
     * @return A constant reference to the field of face unit normals.
     */
    const vectorVector& nf() const;

    /**
     * @brief Get a view of face unit normals for a specific boundary face.
     *
     * @param i The index of the boundary face.
     * @return A view of face unit normals for the specified boundary face.
     */
    View<const Vec3> nf(const localIdx i) const;

    /**
     * @brief Get the field of delta vectors.
     *
     * @return A constant reference to the field of delta vectors.
     */
    const vectorVector& delta() const;

    /**
     * @brief Get a view of delta vectors for a specific boundary face.
     *
     * @param i The index of the boundary face.
     * @return A view of delta vectors for the specified boundary face.
     */
    View<const Vec3> delta(const localIdx i) const;

    /**
     * @brief Get the field of weights.
     *
     * @return A constant reference to the boundary field of weights.
     */
    const scalarVector& weights() const;

    /**
     * @brief Get a view of weights for a specific boundary face.
     *
     * @param i The index of the boundary face.
     * @return A view of weights for the specified boundary face.
     */
    View<const scalar> weights(const localIdx i) const;

    /**
     * @brief Get the field of delta coefficients.
     *
     * @return A constant reference to the field of delta coefficients.
     */
    const scalarVector& deltaCoeffs() const;

    /**
     * @brief Get a view of delta coefficients for a specific boundary face.
     *
     * @param i The index of the boundary face.
     * @return A view of delta coefficients for the specified boundary face.
     */
    View<const scalar> deltaCoeffs(const localIdx i) const;

    /**
     * @brief Get the offset of the boundary faces.
     *
     * @return A constant reference to the offset of the boundary faces.
     */
    // FIXME use Vector here?
    const std::vector<localIdx>& offset() const;


private:

    /**
     * @brief Executor used for computations.
     */
    const Executor exec_;

    /**
     * @brief Vector of face cells.
     *
     * A field with the neighbouring cell of each boundary face.
     */
    labelVector faceCells_;

    /**
     * @brief Vector of face centres.
     */
    vectorVector Cf_;

    /**
     * @brief Vector of face normals.
     */
    vectorVector Cn_;

    /**
     * @brief Vector of face areas normals.
     */
    vectorVector Sf_;

    /**
     * @brief Vector of magnitudes of face areas normals.
     */
    scalarVector magSf_;

    /**
     * @brief Vector of face unit normals.
     */
    vectorVector nf_;

    /**
     * @brief Vector of delta vectors.
     *
     * The delta vector is defined as the vector from the face centre to the
     * cell centre.
     */
    vectorVector delta_;

    /**
     * @brief Vector of weights.
     *
     * The weights are used in cell to face interpolation.
     */
    scalarVector weights_;

    /**
     * @brief Vector of delta coefficients.
     *
     * Vector of cell to face distances.
     */
    scalarVector deltaCoeffs_;

    /**
     * @brief Offset of the boundary faces.
     *
     * The offset is used to access the boundary faces of each boundary.
     */
    // FIXME use Vector here?
    std::vector<localIdx> offset_;
};

} // namespace NeoN
