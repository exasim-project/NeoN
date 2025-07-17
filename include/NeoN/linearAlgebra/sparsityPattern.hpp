// SPDX-FileCopyrightText: 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/array.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::la
{

/* @class SparsityPattern
 * @brief row and column index representation of a mesh
 *
 * This class implements the finite volume 3/5/7 pt stencil specific generation
 * of sparsity patterns from a given unstructured mesh
 *
 */
class SparsityPattern
{
public:

    /* @brief create an SparsityPattern from existing mesh */
    SparsityPattern(const UnstructuredMesh& mesh);

    /* @brief create an "empty" SparsityPattern with a given size  */
    SparsityPattern(Executor exec, localIdx nRows, localIdx nnzs);

    SparsityPattern(
        Array<uint8_t>&& rowOffs,
        Array<uint8_t>&& colIdxs,
        Array<uint8_t>&& ownerOffset,
        Vector<localIdx>&& neighbourOffset,
        Vector<localIdx>&& diagOffset
    );

    /*@brief getter for ownerOffset */
    const Array<uint8_t>& ownerOffset() const;

    /*@brief getter for neighbourOffset */
    const Array<uint8_t>& neighbourOffset() const;

    /*@brief getter for diagOffset */
    const Array<uint8_t>& diagOffset() const;

    /*@brief getter for ownerOffset */
    Array<uint8_t>& ownerOffset();

    /*@brief getter for neighbourOffset */
    Array<uint8_t>& neighbourOffset();

    /*@brief getter for diagOffset */
    Array<uint8_t>& diagOffset();

    /*@brief getter for diagOffset */
    const Executor& exec() const { return exec_; };

    /*@brief getter for colIdxs */
    [[nodiscard]] const Vector<localIdx>& colIdxs() const { return colIdxs_; };

    [[nodiscard]] Vector<localIdx>& colIdxs() { return colIdxs_; };

    /*@brief getter for rowOffs */
    [[nodiscard]] const Vector<localIdx>& rowOffs() const { return rowOffs_; };

    /*@brief getter for rowOffs */
    [[nodiscard]] Vector<localIdx>& rowOffs() { return rowOffs_; };

    [[nodiscard]] localIdx rows() const { return diagOffset_.size(); };

    [[nodiscard]] localIdx nnz() const { return colIdxs_.size(); };

    // TODO add selection mechanism via dictionary later
    static const SparsityPattern& readOrCreate(const UnstructuredMesh& mesh);

private:

    Executor exec_;

    Vector<localIdx> rowOffs_; //! rowOffs map from row to start index in values

    Vector<localIdx> colIdxs_; //!

    Array<uint8_t> ownerOffset_; //! mapping from faceId to lower index in a row

    Array<uint8_t> neighbourOffset_; //! mapping from faceId to upper index in a row

    Array<uint8_t> diagOffset_; //! mapping from faceId to column index in a row
};

SparsityPattern createSparsity(const UnstructuredMesh& mesh);

SparsityPattern updateSparsity(const UnstructuredMesh& mesh, SparsityPattern& in);

} // namespace NeoN::la
