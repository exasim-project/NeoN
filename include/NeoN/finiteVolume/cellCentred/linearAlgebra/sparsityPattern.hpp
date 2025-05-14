// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoN authors

#pragma once


#include "NeoN/core/array.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::finiteVolume::cellCentred
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

    // TODO implement ctor copying all members
    SparsityPattern(const UnstructuredMesh& mesh);

    void update();

    // TODO: rename upperOffset
    const Array<uint8_t>& ownerOffset() const;

    // TODO: rename lowerOffset
    const Array<uint8_t>& neighbourOffset() const;

    const Array<uint8_t>& diagOffset() const;

    const UnstructuredMesh& mesh() const { return mesh_; };

    [[nodiscard]] const Vector<localIdx>& colIdxs() const { return colIdxs_; };

    [[nodiscard]] const Vector<localIdx>& rowOffs() const { return rowOffs_; };

    [[nodiscard]] localIdx rows() const { return diagOffset_.size(); };

    [[nodiscard]] localIdx nnz() const { return colIdxs_.size(); };

    // add selection mechanism via dictionary later
    static const std::shared_ptr<SparsityPattern> readOrCreate(const UnstructuredMesh& mesh);

private:

    const UnstructuredMesh& mesh_;

    Vector<localIdx> rowOffs_; //! rowOffs map from row to start index in values

    Vector<localIdx> colIdxs_; //!

    Array<uint8_t> ownerOffset_; //! mapping from faceId to lower index in a row

    Array<uint8_t> neighbourOffset_; //! mapping from faceId to upper index in a row

    Array<uint8_t> diagOffset_; //! mapping from faceId to column index in a row
};

} // namespace NeoN::finiteVolume::cellCentred
