// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/io/cgnsMeshWriter.hpp"
#include "NeoN/mesh/unstructured/io/meshConverter.hpp"

#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/vector/vector.hpp"
#include "NeoN/core/vector/vectorTypeDefs.hpp"
#include "NeoN/mesh/unstructured/boundaryMesh.hpp"

// clang-format off
#include <vtk_cgns.h>
#include VTK_CGNS(cgnslib.h)
// clang-format on

#include <map>
#include <stdexcept>
#include <string>
#include <vector>


namespace NeoN::io
{

namespace
{

// Map VTK cell type ID to CGNS element type
CGNS_ENUMT(ElementType_t) vtkTypeToCgns(int vtkType)
{
    constexpr int VTK_TETRA_TYPE = 10;
    constexpr int VTK_HEXAHEDRON_TYPE = 12;
    constexpr int VTK_WEDGE_TYPE = 13;
    constexpr int VTK_PYRAMID_TYPE = 14;

    switch (vtkType)
    {
    case VTK_TETRA_TYPE:
        return CGNS_ENUMV(TETRA_4);
    case VTK_HEXAHEDRON_TYPE:
        return CGNS_ENUMV(HEXA_8);
    case VTK_WEDGE_TYPE:
        return CGNS_ENUMV(PENTA_6);
    case VTK_PYRAMID_TYPE:
        return CGNS_ENUMV(PYRA_5);
    default:
        return CGNS_ENUMV(TETRA_4);
    }
}

// Convert 0-based node indices to 1-based cgsize_t for CGNS
std::vector<cgsize_t> toCgnsNodes(const std::vector<localIdx>& nodes)
{
    std::vector<cgsize_t> result;
    result.reserve(nodes.size());
    for (localIdx n : nodes)
    {
        result.push_back(static_cast<cgsize_t>(n + 1));
    }
    return result;
}

} // anonymous namespace


void writeCgns(const UnstructuredMesh& mesh, const std::string& filePath)
{
    localIdx nCells = mesh.nCells();
    localIdx nInternalFaces = mesh.nInternalFaces();
    localIdx nBoundaryFaces = mesh.nBoundaryFaces();
    localIdx nFaces = mesh.nFaces();

    // Copy all data to host
    auto hostPoints = mesh.points().copyToHost();
    auto hostFaceOwner = mesh.faceOwner().copyToHost();
    auto hostFaceNeighbour = mesh.faceNeighbour().copyToHost();
    localIdx nPoints = hostPoints.size();

    // Retrieve face node connectivity from stencilDB if stored during read
    std::vector<std::vector<localIdx>> faceNodes;

    if (mesh.stencilDB().contains("io::faceNodes"))
    {
        auto& stored = mesh.stencilDB().get<std::shared_ptr<std::vector<std::vector<localIdx>>>>(
            "io::faceNodes"
        );
        faceNodes = *stored;
    }
    else
    {
        throw std::runtime_error("writeCgns: face node connectivity not available in stencilDB. "
                                 "Only meshes created by readCgns can be written.");
    }

    // Convert faceOwner/faceNeighbour to std::vector
    std::vector<label> faceOwnerVec(static_cast<std::size_t>(nFaces));
    for (localIdx i = 0; i < nFaces; ++i)
    {
        faceOwnerVec[static_cast<std::size_t>(i)] = hostFaceOwner.view()[i];
    }
    std::vector<label> faceNeighbourVec(static_cast<std::size_t>(nInternalFaces));
    for (localIdx i = 0; i < nInternalFaces; ++i)
    {
        faceNeighbourVec[static_cast<std::size_t>(i)] = hostFaceNeighbour.view()[i];
    }

    // Rebuild cell info via meshConverter
    auto cells =
        rebuildCellInfo(faceOwnerVec, faceNeighbourVec, faceNodes, nCells, nInternalFaces, nFaces);

    // Open CGNS file for writing
    int fn = 0;
    if (cg_open(filePath.c_str(), CG_MODE_WRITE, &fn) != CG_OK)
    {
        throw std::runtime_error("Failed to open CGNS file for writing: " + filePath);
    }

    // Write base
    int baseIdx = 0;
    int cellDim = 3;
    int physDim = 3;
    if (cg_base_write(fn, "NeoNBase", cellDim, physDim, &baseIdx) != CG_OK)
    {
        cg_close(fn);
        throw std::runtime_error("Failed to write CGNS base");
    }

    // Write zone
    cgsize_t sizes[3] = {
        static_cast<cgsize_t>(nPoints),
        static_cast<cgsize_t>(nCells),
        0 // boundary vertex size (0 for unstructured)
    };
    int zoneIdx = 0;
    if (cg_zone_write(fn, baseIdx, "Zone1", sizes, CGNS_ENUMV(Unstructured), &zoneIdx) != CG_OK)
    {
        cg_close(fn);
        throw std::runtime_error("Failed to write CGNS zone");
    }

    // Write coordinates
    std::vector<double> coordX(static_cast<std::size_t>(nPoints));
    std::vector<double> coordY(static_cast<std::size_t>(nPoints));
    std::vector<double> coordZ(static_cast<std::size_t>(nPoints));
    for (localIdx i = 0; i < nPoints; ++i)
    {
        auto si = static_cast<std::size_t>(i);
        coordX[si] = static_cast<double>(hostPoints.view()[i][0]);
        coordY[si] = static_cast<double>(hostPoints.view()[i][1]);
        coordZ[si] = static_cast<double>(hostPoints.view()[i][2]);
    }

    int coordIdx = 0;
    cg_coord_write(
        fn, baseIdx, zoneIdx, CGNS_ENUMV(RealDouble), "CoordinateX", coordX.data(), &coordIdx
    );
    cg_coord_write(
        fn, baseIdx, zoneIdx, CGNS_ENUMV(RealDouble), "CoordinateY", coordY.data(), &coordIdx
    );
    cg_coord_write(
        fn, baseIdx, zoneIdx, CGNS_ENUMV(RealDouble), "CoordinateZ", coordZ.data(), &coordIdx
    );

    // Group cells by CGNS element type
    std::map<CGNS_ENUMT(ElementType_t), std::vector<std::size_t>> cellsByType;
    for (std::size_t c = 0; c < cells.size(); ++c)
    {
        cellsByType[vtkTypeToCgns(cells[c].cellType)].push_back(c);
    }

    // Write volume element sections
    cgsize_t elemStart = 1;
    for (auto& [elemType, cellIndices] : cellsByType)
    {
        int nodesPerElem = 0;
        std::string secName;

        switch (elemType)
        {
        case CGNS_ENUMV(TETRA_4):
            nodesPerElem = 4;
            secName = "Tetra";
            break;
        case CGNS_ENUMV(HEXA_8):
            nodesPerElem = 8;
            secName = "Hexa";
            break;
        case CGNS_ENUMV(PENTA_6):
            nodesPerElem = 6;
            secName = "Penta";
            break;
        case CGNS_ENUMV(PYRA_5):
            nodesPerElem = 5;
            secName = "Pyra";
            break;
        default:
            continue;
        }

        std::vector<cgsize_t> connectivity;
        connectivity.reserve(cellIndices.size() * static_cast<std::size_t>(nodesPerElem));

        for (std::size_t ci : cellIndices)
        {
            std::vector<cgsize_t> orderedNodes;
            if (elemType == CGNS_ENUMV(TETRA_4))
            {
                orderedNodes = toCgnsNodes(orderTetNodes(cells[ci]));
            }
            else if (elemType == CGNS_ENUMV(HEXA_8))
            {
                orderedNodes = toCgnsNodes(orderHexNodes(cells[ci]));
            }
            else if (elemType == CGNS_ENUMV(PYRA_5))
            {
                orderedNodes = toCgnsNodes(orderPyramidNodes(cells[ci]));
            }
            else if (elemType == CGNS_ENUMV(PENTA_6))
            {
                orderedNodes = toCgnsNodes(orderWedgeNodes(cells[ci]));
            }
            else
            {
                orderedNodes = toCgnsNodes(cells[ci].nodeIds);
            }
            connectivity.insert(connectivity.end(), orderedNodes.begin(), orderedNodes.end());
        }

        cgsize_t elemEnd = elemStart + static_cast<cgsize_t>(cellIndices.size()) - 1;
        int secIdx = 0;
        cg_section_write(
            fn,
            baseIdx,
            zoneIdx,
            secName.c_str(),
            elemType,
            elemStart,
            elemEnd,
            0,
            connectivity.data(),
            &secIdx
        );
        elemStart = elemEnd + 1;
    }

    // Write boundary face element sections (one per patch)
    auto const& offset = mesh.boundaryMesh().offset();
    localIdx nBoundaries = mesh.nBoundaries();

    for (localIdx b = 0; b < nBoundaries; ++b)
    {
        localIdx patchStart = offset[static_cast<std::size_t>(b)];
        localIdx patchEnd = offset[static_cast<std::size_t>(b + 1)];
        localIdx patchSize = patchEnd - patchStart;

        if (patchSize == 0) continue;

        // Determine face element type from first face
        auto firstFaceIdx = static_cast<std::size_t>(nInternalFaces + patchStart);
        localIdx nFaceNodes = static_cast<localIdx>(faceNodes[firstFaceIdx].size());

        CGNS_ENUMT(ElementType_t) faceType;
        if (nFaceNodes == 3) faceType = CGNS_ENUMV(TRI_3);
        else if (nFaceNodes == 4)
            faceType = CGNS_ENUMV(QUAD_4);
        else
            faceType = CGNS_ENUMV(TRI_3); // fallback

        std::vector<cgsize_t> faceConn;
        faceConn.reserve(static_cast<std::size_t>(patchSize * nFaceNodes));

        for (localIdx i = patchStart; i < patchEnd; ++i)
        {
            auto fi = static_cast<std::size_t>(nInternalFaces + i);
            for (localIdx n : faceNodes[fi])
            {
                faceConn.push_back(static_cast<cgsize_t>(n + 1)); // 1-based
            }
        }

        // Retrieve patch name from stencilDB if available
        std::string patchName = "patch_" + std::to_string(b);
        if (mesh.stencilDB().contains("io::patchNames"))
        {
            auto& names =
                mesh.stencilDB().get<std::shared_ptr<std::vector<std::string>>>("io::patchNames");
            if (static_cast<std::size_t>(b) < names->size())
            {
                patchName = (*names)[static_cast<std::size_t>(b)];
            }
        }

        cgsize_t secStart = elemStart;
        cgsize_t secEnd = elemStart + static_cast<cgsize_t>(patchSize) - 1;
        int secIdx = 0;
        cg_section_write(
            fn,
            baseIdx,
            zoneIdx,
            patchName.c_str(),
            faceType,
            secStart,
            secEnd,
            0,
            faceConn.data(),
            &secIdx
        );

        // Write BC_t node for this patch
        cgsize_t bcRange[2] = {secStart, secEnd};
        int bcIdx = 0;
        cg_boco_write(
            fn,
            baseIdx,
            zoneIdx,
            patchName.c_str(),
            CGNS_ENUMV(BCWall),
            CGNS_ENUMV(PointRange),
            2,
            bcRange,
            &bcIdx
        );

        // Set GridLocation to CellCenter (element-based)
        cg_goto(fn, baseIdx, "Zone_t", zoneIdx, "ZoneBC_t", 1, "BC_t", bcIdx, "end");
        cg_gridlocation_write(CGNS_ENUMV(CellCenter));

        elemStart = secEnd + 1;
    }

    cg_close(fn);
}


} // namespace NeoN::io
