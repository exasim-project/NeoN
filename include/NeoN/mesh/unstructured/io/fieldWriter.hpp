// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"

#include <functional>
#include <string>
#include <variant>
#include <vector>

namespace NeoN::io
{

namespace fvcc = NeoN::finiteVolume::cellCentred;

/**
 * @brief Accumulates references to VolumeFields for a single write pass.
 *
 * Building VTK topology is expensive. FieldSet lets you add multiple fields
 * and write them all in one pass (one topology build, multiple AddArray calls).
 *
 * Example:
 * @code
 *   NeoN::io::FieldSet fs;
 *   fs.add(pressure).add(velocity);
 *   NeoN::io::writeVtm(mesh, fs, "output.vtm");
 * @endcode
 */
class FieldSet
{
public:

    /** @brief Add a scalar VolumeField. Returns *this for chaining. */
    FieldSet& add(const fvcc::VolumeField<scalar>& field);

    /** @brief Add a Vec3 VolumeField. Returns *this for chaining. */
    FieldSet& add(const fvcc::VolumeField<Vec3>& field);

private:

    using AnyField = std::variant<
        std::reference_wrapper<const fvcc::VolumeField<scalar>>,
        std::reference_wrapper<const fvcc::VolumeField<Vec3>>>;
    std::vector<AnyField> fields_;

    friend void writeVtm(const UnstructuredMesh&, const FieldSet&, const std::string&);
    friend void writeVtkHdf(const UnstructuredMesh&, const FieldSet&, const std::string&);
};

// --- VTM (multiblock XML) ---

/**
 * @brief Write an UnstructuredMesh with a scalar VolumeField to a VTM (multi-block) file.
 *
 * Internal field data is written as CellData on the volume grid block.
 * Boundary field data is written as CellData on each boundary patch block.
 *
 * @param mesh The mesh to write.
 * @param field The scalar volume field to include.
 * @param filePath Output path (must end in .vtm).
 */
void writeVtm(
    const UnstructuredMesh& mesh,
    const fvcc::VolumeField<scalar>& field,
    const std::string& filePath
);

/**
 * @brief Write an UnstructuredMesh with a Vec3 VolumeField to a VTM (multi-block) file.
 *
 * Internal field data is written as CellData on the volume grid block.
 * Boundary field data is written as CellData on each boundary patch block.
 *
 * @param mesh The mesh to write.
 * @param field The Vec3 volume field to include.
 * @param filePath Output path (must end in .vtm).
 */
void writeVtm(
    const UnstructuredMesh& mesh, const fvcc::VolumeField<Vec3>& field, const std::string& filePath
);

// --- VTK HDF (binary HDF5) ---

/**
 * @brief Write an UnstructuredMesh with a scalar VolumeField to a VTK HDF file.
 *
 * Internal field data is written as CellData on the volume partition.
 * Boundary field data is written as CellData on each boundary patch partition.
 *
 * @param mesh The mesh to write.
 * @param field The scalar volume field to include.
 * @param filePath Output path (must end in .vtkhdf).
 */
void writeVtkHdf(
    const UnstructuredMesh& mesh,
    const fvcc::VolumeField<scalar>& field,
    const std::string& filePath
);

/**
 * @brief Write an UnstructuredMesh with a Vec3 VolumeField to a VTK HDF file.
 *
 * Internal field data is written as CellData on the volume partition.
 * Boundary field data is written as CellData on each boundary patch partition.
 *
 * @param mesh The mesh to write.
 * @param field The Vec3 volume field to include.
 * @param filePath Output path (must end in .vtkhdf).
 */
void writeVtkHdf(
    const UnstructuredMesh& mesh, const fvcc::VolumeField<Vec3>& field, const std::string& filePath
);

// --- Multi-field overloads (FieldSet) ---

/**
 * @brief Write an UnstructuredMesh with all fields in a FieldSet to a VTM file.
 *
 * Topology is built once; all fields are attached as CellData arrays.
 *
 * @param mesh The mesh to write.
 * @param fields FieldSet containing the fields to include.
 * @param filePath Output path (must end in .vtm).
 */
void writeVtm(const UnstructuredMesh& mesh, const FieldSet& fields, const std::string& filePath);

/**
 * @brief Write an UnstructuredMesh with all fields in a FieldSet to a VTK HDF file.
 *
 * Topology is built once; all fields are attached as CellData arrays.
 *
 * @param mesh The mesh to write.
 * @param fields FieldSet containing the fields to include.
 * @param filePath Output path (must end in .vtkhdf).
 */
void writeVtkHdf(const UnstructuredMesh& mesh, const FieldSet& fields, const std::string& filePath);

} // namespace NeoN::io
