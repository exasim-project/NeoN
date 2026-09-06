// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <string>

#include "catch2_common.hpp"

#include "NeoN/NeoN.hpp"

using NeoN::scalar;
using NeoN::localIdx;
using NeoN::Vector;
using NeoN::la::LinearSystem;
using NeoN::la::CSRMatrix;

TEMPLATE_TEST_CASE("matrixIterator", "[template]", NeoN::scalar)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    Vector<scalar> values(exec, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
    Vector<localIdx> colIdx(exec, {0, 1, 2, 0, 1, 2, 0, 1, 2});
    Vector<localIdx> rowOffs(exec, {0, 3, 6, 9});
    CSRMatrix<scalar, localIdx> csrMatrix(values, colIdx, rowOffs, {3, 3});

    SECTION("Construct with MeshCellIterator " + execName)
    {
        auto nCells = 5;
        auto mesh = create1DUniformMesh(exec, nCells);
        auto [sp, mi] = NeoN::la::createSparsityPatternFaceToMatrixAddress<
            NeoN::la::CsrSparsityPattern<NeoN::localIdx>>(mesh);
        auto cellIterator = std::make_shared<NeoN::la::CellBasedIterator>();
        cellIterator->setComputeCellBasedData(mesh, sp, mi);

        auto cellBasedData = cellIterator->getCellBasedData();

        auto cellFacesIdx = cellBasedData->cellFaces.values().copyToHost();
        auto cellFacesOffsets = cellBasedData->cellFaces.segments().copyToHost();
        auto faceNeighbour = cellBasedData->faceNeighbour.copyToHost();
        auto faceSign = cellBasedData->faceSign.copyToHost();
        auto matrixColumnIdx = cellBasedData->matrixColumnIdx.copyToHost();

        // [ | 0 | |  1  | |   2 | |   3 | |  4 | ]
        auto expFaceCellIdx =
            NeoN::Vector<localIdx>(NeoN::SerialExecutor(), {0, 0, 1, 1, 2, 2, 3, 3});
        auto expFaceOffsets = NeoN::Vector<localIdx>(NeoN::SerialExecutor(), {0, 1, 3, 5, 7, 8});
        auto expFaceNeighbour =
            NeoN::Vector<localIdx>(NeoN::SerialExecutor(), {1, 0, 2, 1, 3, 2, 4, 3});
        auto expFaceSign =
            NeoN::Vector<scalar>(NeoN::SerialExecutor(), {1, -1, 1, -1, 1, -1, 1, -1});
        auto expMatrixColumnIdx =
            NeoN::Vector<localIdx>(NeoN::SerialExecutor(), {1, 2, 4, 5, 7, 8, 10, 11});

        REQUIRE_THAT(faceSign, Equals(expFaceSign, Approx {1e-10}));
        REQUIRE_THAT(faceNeighbour, Equals(expFaceNeighbour, EqualInt {}));
        REQUIRE_THAT(cellFacesOffsets, Equals(expFaceOffsets, EqualInt {}));
        REQUIRE_THAT(cellFacesIdx, Equals(expFaceCellIdx, EqualInt {}));
        REQUIRE_THAT(matrixColumnIdx, Equals(expMatrixColumnIdx, EqualInt {}));
    }

    SECTION("Can construct linearSystem with MeshCellIterator " + execName)
    {
        auto nCells = 5;
        auto mesh = create1DUniformMesh(exec, nCells);
        auto cellIterator = std::make_shared<NeoN::la::CellBasedIterator>();
        auto ls = createEmptyLinearSystem<TestType>(mesh, cellIterator);

        REQUIRE(ls.getMeshIterator()->name() == "CellBased");
    }
}
