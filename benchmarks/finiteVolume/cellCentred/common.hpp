// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/NeoN.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

template<typename ValueType>
struct CreateVector
{
    std::string name;
    const NeoN::UnstructuredMesh& mesh;
    std::int64_t timeIndex = 0;
    std::int64_t iterationIndex = 0;
    std::int64_t subCycleIndex = 0;

    NeoN::Document operator()(NeoN::Database& db)
    {
        auto bcs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<ValueType>>(mesh);
        NeoN::Field<ValueType> domainVector(
            mesh.exec(),
            NeoN::Vector<ValueType>(mesh.exec(), mesh.nCells(), NeoN::one<ValueType>()),
            mesh.boundaryMesh().offset()
        );
        fvcc::VolumeField<ValueType> vf(mesh.exec(), name, mesh, domainVector, bcs, db, "", "");

        return NeoN::Document(
            {{"name", vf.name},
             {"timeIndex", timeIndex},
             {"iterationIndex", iterationIndex},
             {"subCycleIndex", subCycleIndex},
             {"field", vf}},
            fvcc::validateVectorDoc
        );
    }
};
