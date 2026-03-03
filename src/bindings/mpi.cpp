// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

#ifdef NF_WITH_MPI_SUPPORT
#include "NeoN/core/mpi/environment.hpp"
#include "NeoN/mesh/unstructured/communicator.hpp"
#endif

#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace NeoN::bindings
{

#ifdef NF_WITH_MPI_SUPPORT

Communicator
buildCommunicatorFromMesh(const UnstructuredMesh& mesh, const mpi::MPIEnvironment& mpiEnviron)
{
    auto& nPartsPtr = mesh.stencilDB().get<std::shared_ptr<int>>("partition::nParts");
    int nParts = *nPartsPtr;

    auto& sendData = *mesh.stencilDB().get<std::shared_ptr<std::vector<std::vector<localIdx>>>>(
        "partition::commSendMap"
    );
    auto& recvData = *mesh.stencilDB().get<std::shared_ptr<std::vector<std::vector<localIdx>>>>(
        "partition::commReceiveMap"
    );

    // Build CommMap sized to sizeRank (Communicator asserts map size == sizeRank).
    // For ranks beyond nParts, leave empty.
    CommMap sendMap(mpiEnviron.sizeRank()), receiveMap(mpiEnviron.sizeRank());
    for (int r = 0; r < nParts && r < static_cast<int>(mpiEnviron.sizeRank()); ++r)
    {
        for (auto idx : sendData[static_cast<std::size_t>(r)])
            sendMap[static_cast<std::size_t>(r)].push_back(
                NodeCommMap {.local_idx = static_cast<label>(idx)}
            );
        for (auto idx : recvData[static_cast<std::size_t>(r)])
            receiveMap[static_cast<std::size_t>(r)].push_back(
                NodeCommMap {.local_idx = static_cast<label>(idx)}
            );
    }

    return Communicator(mpiEnviron, sendMap, receiveMap);
}

#endif

void registerMPI(nb::module_& m)
{
#ifdef NF_WITH_MPI_SUPPORT
    // MPIEnvironment
    nb::class_<mpi::MPIEnvironment>(m, "MPIEnvironment", "MPI environment wrapper")
        .def(nb::init<>(), "Create MPIEnvironment for MPI_COMM_WORLD")
        .def("rank", &mpi::MPIEnvironment::rank, "Get the rank of this process")
        .def("size", &mpi::MPIEnvironment::sizeRank, "Get the total number of ranks");

    // Communicator
    nb::class_<Communicator>(m, "Communicator", "MPI communicator for ghost cell sync")
        .def(
            "start_comm",
            &Communicator::startComm<scalar>,
            "field"_a,
            "comm_name"_a,
            "Start non-blocking scalar ghost cell communication"
        )
        .def(
            "is_complete",
            &Communicator::isComplete,
            "comm_name"_a,
            "Check if communication is complete"
        )
        .def(
            "finalise_comm",
            &Communicator::finaliseComm<scalar>,
            "field"_a,
            "comm_name"_a,
            "Finalize scalar ghost cell communication"
        );

    // build_communicator helper
    m.def(
        "build_communicator",
        &buildCommunicatorFromMesh,
        "mesh"_a,
        "mpi_env"_a,
        "Build a Communicator from a partitioned sub-mesh's stencilDB metadata."
    );
#endif
}

} // namespace NeoN::bindings
