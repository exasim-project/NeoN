// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "catch2_common.hpp"

#include "../dsl/common.hpp"

namespace fvcc = NeoN::finiteVolume::cellCentred;

namespace NeoN
{

// 2-rank MPI test for BoundaryData staging-buffer lifecycle.
//
// This test exercises the host-staging path under NEON_FORCE_HOST_BUFFER=1 and
// asserts the three observable behaviours the persistent-pool refactor must satisfy:
//   [ALLOC-01] pointer/capacity stability across send/recv rounds
//   [ALLOC-02] staging size == patchSize, not nBoundaryFaces
//   [ALLOC-03] pool reset on operator=, followed by lazy rebuild
//
// For a 2-rank 1D partition with nCells=8 (4 per rank):
//   Rank 0: physical boundary at patchId 0 (rangeStart 0, size 1),
//           processor boundary at patchId 1 (rangeStart 1, size 1)
//   Rank 1: physical boundary at patchId 0 (rangeStart 0, size 1),
//           processor boundary at patchId 1 (rangeStart 1, size 1)
//
// Expected values after correctBoundaryConditions():
//   Rank 0: proc ghost = rank 1's first cell = 5.0
//   Rank 1: proc ghost = rank 0's last cell  = 4.0

TEST_CASE("BoundaryData persistent staging buffer host path", "[ALLOC-01][ALLOC-02]")
{
    // CPU executor: this test reads std::vector capacity from the host staging path.
    // Under NEON_FORCE_HOST_BUFFER=1 the host path is taken regardless of hardware.
    const Executor exec = CPUExecutor {};
    mpi::Environment mpiEnviron;

    const auto nCells = 8; // 2 ranks x 4 cells per rank
    auto meshGlobal = create1DUniformMesh(exec, nCells);
    auto volBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(meshGlobal);
    auto uVals = std::vector<scalar> {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    auto u = fvcc::VolumeField<scalar>(exec, "U", meshGlobal, Vector<scalar>(exec, uVals), volBCs);

    auto meshPart = create1DUniformMeshPart(exec, nCells / mpiEnviron.sizeRank());
    auto uPart = detail::oneDPartitionField(u, meshPart, mpiEnviron);

    // Round 1: post sends/recvs via correctBoundaryConditions, drain via value()
    uPart.correctBoundaryConditions();
    auto& bdVal1 = uPart.boundaryData().value(); // triggers waitAll -- round 1 complete
    (void)bdVal1;

    // The proc patch is patchId 1 (physical patch is patchId 0).
    const localIdx procPatchId = 1;
    const auto [rangeStart, rangeEnd] = uPart.boundaryData().range(procPatchId);
    const localIdx patchSize = rangeEnd - rangeStart;

    // Pool is now keyed by neighbourRank (not rangeStart).
    // Obtain the neighbour rank for the proc patch so we can query the pool.
    const int procPatchNeighbourRank =
        static_cast<int>(meshPart.boundaryMesh().neighbourRankForRange({rangeStart, rangeEnd}));

    // Record the staging buffer address and capacity after round 1.
    const scalar* ptr1 = uPart.boundaryData().sendBufPtrForTest(procPatchNeighbourRank);
    const std::size_t cap1 = uPart.boundaryData().sendBufCapForTest(procPatchNeighbourRank);
    const std::size_t sz1 = uPart.boundaryData().sendBufSizeForTest(procPatchNeighbourRank);

    REQUIRE(ptr1 != nullptr); // host staging buffer must exist after round 1

    // Round 2: run another send/recv cycle
    uPart.correctBoundaryConditions();
    uPart.boundaryData().value(); // drain round 2

    // [ALLOC-01]: pointer and capacity must be unchanged after round 2
    // (current code reallocated the staging buffer every round — this FAILS RED pre-refactor)
    REQUIRE(uPart.boundaryData().sendBufPtrForTest(procPatchNeighbourRank) == ptr1);
    REQUIRE(uPart.boundaryData().sendBufCapForTest(procPatchNeighbourRank) == cap1);

    // [ALLOC-02]: staging size must equal patchSize, not nBoundaryFaces
    // (current code uses resize(patchSize) but commBuffers_.clear() destroys and rebuilds)
    REQUIRE(static_cast<localIdx>(sz1) == patchSize);

    // Verify the received neighbour value is correct (functional correctness check)
    SECTION_IF(mpiEnviron.rank() == 0, "Rank 0 received rank 1 first cell value")
    {
        auto uPartBoundExp = std::vector<scalar> {0.0, 5.0};
        REQUIRE_THAT(uPart.boundaryData().value(), Equals(uPartBoundExp));
    }
    SECTION_IF(mpiEnviron.rank() == 1, "Rank 1 received rank 0 last cell value")
    {
        auto uPartBoundExp = std::vector<scalar> {0.0, 4.0};
        REQUIRE_THAT(uPart.boundaryData().value(), Equals(uPartBoundExp));
    }
}

TEST_CASE("BoundaryData operator= resets staging pool", "[ALLOC-03]")
{
    const Executor exec = CPUExecutor {};
    mpi::Environment mpiEnviron;

    const auto nCells = 8;
    auto meshGlobal = create1DUniformMesh(exec, nCells);
    auto volBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<scalar>>(meshGlobal);
    auto uVals = std::vector<scalar> {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    auto u = fvcc::VolumeField<scalar>(exec, "U", meshGlobal, Vector<scalar>(exec, uVals), volBCs);

    auto meshPart = create1DUniformMeshPart(exec, nCells / mpiEnviron.sizeRank());
    auto uPart = detail::oneDPartitionField(u, meshPart, mpiEnviron);

    // Round 1: populate the staging pool
    uPart.correctBoundaryConditions();
    uPart.boundaryData().value(); // drain round 1

    // Pool must be non-empty after round 1
    REQUIRE(uPart.boundaryData().poolSizeForTest() >= 1);

    // Build a second partitioned field (same mesh — the observable is pool reset, not size change)
    auto uPart2 = detail::oneDPartitionField(u, meshPart, mpiEnviron);
    uPart2.correctBoundaryConditions();
    uPart2.boundaryData().value(); // drain round 1 for uPart2

    // Assign uPart2's BoundaryData into uPart1's BoundaryData.
    // [ALLOC-03]: operator= must reset the staging pool to empty
    // (current code does NOT clear commBuffers_ on assignment — this FAILS RED pre-refactor)
    uPart.boundaryData() = uPart2.boundaryData();
    REQUIRE(uPart.boundaryData().poolSizeForTest() == 0);

    // After assignment, a fresh send/recv round must succeed and rebuild the pool
    uPart.correctBoundaryConditions();
    uPart.boundaryData().value(); // drain post-assignment round

    REQUIRE(uPart.boundaryData().poolSizeForTest() >= 1);
}

} // namespace NeoN
