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
        labelField faceCells,
        vectorField cf,
        vectorField cn,
        vectorField sf,
        scalarField magSf,
        vectorField nf,
        vectorField delta,
        scalarField weights,
        scalarField deltaCoeffs,
        std::vector<localIdx> offset
    );


    /**
     * @brief Get the field of face cells.
     *
     * @return A constant reference to the field of face cells.
     */
    const labelField& faceCells() const;

    // TODO either dont mix return types, ie dont use view and Field
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
    const vectorField& cf() const;

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
    const vectorField& cn() const;

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
    const vectorField& sf() const;

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
    const scalarField& magSf() const;

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
    const vectorField& nf() const;

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
    const vectorField& delta() const;

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
    const scalarField& weights() const;

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
    const scalarField& deltaCoeffs() const;

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
    const std::vector<localIdx>& offset() const;


private:

    /**
     * @brief Executor used for computations.
     */
    const Executor exec_;

    /**
     * @brief Field of face cells.
     *
     * A field with the neighbouring cell of each boundary face.
     */
    labelField faceCells_;

    /**
     * @brief Field of face centres.
     */
    vectorField Cf_;

    /**
     * @brief Field of face normals.
     */
    vectorField Cn_;

    /**
     * @brief Field of face areas normals.
     */
    vectorField Sf_;

    /**
     * @brief Field of magnitudes of face areas normals.
     */
    scalarField magSf_;

    /**
     * @brief Field of face unit normals.
     */
    vectorField nf_;

    /**
     * @brief Field of delta vectors.
     *
     * The delta vector is defined as the vector from the face centre to the
     * cell centre.
     */
    vectorField delta_;

    /**
     * @brief Field of weights.
     *
     * The weights are used in cell to face interpolation.
     */
    scalarField weights_;

    /**
     * @brief Field of delta coefficients.
     *
     * Field of cell to face distances.
     */
    scalarField deltaCoeffs_;

    /**
     * @brief Offset of the boundary faces.
     *
     * The offset is used to access the boundary faces of each boundary.
     */
    std::vector<localIdx> offset_;
};

} // namespace NeoN
