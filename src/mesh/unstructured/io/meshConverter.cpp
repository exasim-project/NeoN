// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "NeoN/mesh/unstructured/io/meshConverter.hpp"
#include "NeoN/mesh/unstructured/boundaryMesh.hpp"
#include "NeoN/core/primitives/label.hpp"
#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/vector/vector.hpp"

#include <vtkCompositeDataSet.h>
#include <vtkInformation.h>
#include <vtkMultiBlockDataSet.h>

#include <cmath>
#include <vector>


namespace NeoN::io
{

BoundaryMesh buildBoundaryMesh(
    const Executor& exec,
    const Vector<localIdx>& faceOwner,
    const Vector<Vec3>& faceCenters,
    const Vector<Vec3>& cellCenters,
    const Vector<Vec3>& faceAreas,
    const Vector<scalar>& magFaceAreas,
    localIdx nInternalFaces,
    localIdx nBoundaryFaces,
    const std::vector<localIdx>& patchOffsets
)
{
    auto owView = faceOwner.view();
    auto hFaceCenters = faceCenters.view();
    auto hCellCenters = cellCenters.view();
    auto hFaceAreas = faceAreas.view();
    auto hMagFaceAreas = magFaceAreas.view();

    std::vector<label> bndFaceCells(static_cast<std::size_t>(nBoundaryFaces));
    std::vector<Vec3> bndCf(static_cast<std::size_t>(nBoundaryFaces));
    std::vector<Vec3> bndCn(static_cast<std::size_t>(nBoundaryFaces));
    std::vector<Vec3> bndSf(static_cast<std::size_t>(nBoundaryFaces));
    std::vector<scalar> bndMagSf(static_cast<std::size_t>(nBoundaryFaces));
    std::vector<Vec3> bndNf(static_cast<std::size_t>(nBoundaryFaces));
    std::vector<Vec3> bndDelta(static_cast<std::size_t>(nBoundaryFaces));
    std::vector<scalar> bndWeights(static_cast<std::size_t>(nBoundaryFaces));
    std::vector<scalar> bndDeltaCoeffs(static_cast<std::size_t>(nBoundaryFaces));

    for (localIdx i = 0; i < nBoundaryFaces; ++i)
    {
        const auto bi = static_cast<std::size_t>(i);
        const localIdx fi = nInternalFaces + i;
        const localIdx ownerCell = owView[fi];

        bndFaceCells[bi] = static_cast<label>(ownerCell);
        bndCf[bi] = hFaceCenters[fi];
        bndCn[bi] = hCellCenters[ownerCell];
        bndSf[bi] = hFaceAreas[fi];
        bndMagSf[bi] = hMagFaceAreas[fi];
        bndNf[bi] =
            (bndMagSf[bi] > 1e-30) ? bndSf[bi] * (1.0 / bndMagSf[bi]) : Vec3 {0.0, 0.0, 0.0};
        bndDelta[bi] = bndCf[bi] - bndCn[bi];
        const scalar magDelta = mag(bndDelta[bi]);
        bndDeltaCoeffs[bi] = (magDelta > 1e-30) ? 1.0 / magDelta : 0.0;
        bndWeights[bi] = 1.0;
    }

    // Non-distributed mesh: no processor patches or neighbour ranks.
    return BoundaryMesh(
        exec,
        labelVector(exec, bndFaceCells),
        vectorVector(exec, bndCf),
        vectorVector(exec, bndCn),
        vectorVector(exec, bndSf),
        scalarVector(exec, bndMagSf),
        vectorVector(exec, bndNf),
        vectorVector(exec, bndDelta),
        scalarVector(exec, bndWeights),
        scalarVector(exec, bndDeltaCoeffs),
        patchOffsets,
        0,
        {}
    );
}


std::vector<std::string> multiBlockPatchNames(vtkMultiBlockDataSet* boundary)
{
    std::vector<std::string> names;
    if (!boundary)
    {
        return names;
    }
    for (unsigned int i = 0; i < boundary->GetNumberOfBlocks(); ++i)
    {
        if (boundary->HasMetaData(i) && boundary->GetMetaData(i)->Has(vtkCompositeDataSet::NAME()))
        {
            names.emplace_back(boundary->GetMetaData(i)->Get(vtkCompositeDataSet::NAME()));
        }
        else
        {
            names.emplace_back("patch_" + std::to_string(i));
        }
    }
    return names;
}

} // namespace NeoN::io
