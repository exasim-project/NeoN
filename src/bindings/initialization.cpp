// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "NeoN/core/initialization.hpp"
#ifdef NF_WITH_MPI_SUPPORT
#include "NeoN/core/mpi/environment.hpp"
#endif

#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace NeoN::bindings
{

void registerInitialization(nanobind::module_& m)
{
    m.def(
        "initialize",
        [](nb::list args)
        {
            // Convert Python list to argc/argv
            std::vector<std::string> args_vec;
            for (auto item : args)
            {
                args_vec.push_back(nb::cast<std::string>(item));
            }

            // Create char* array for C-style argv
            std::vector<char*> argv;
            for (auto& arg : args_vec)
            {
                argv.push_back(const_cast<char*>(arg.c_str()));
            }

            int argc = static_cast<int>(argv.size());
            NeoN::initialize(argc, argv.data());
        },
        "args"_a,
        R"doc(
              Initialize NeoN with command line arguments.

              This function initializes the Kokkos execution backend and sets up
              default logging patterns for NeoN.

              Parameters:
                  args: List of command line arguments (similar to sys.argv)

              Example:
                  import sys
                  import neon
                  neon.initialize(sys.argv)
          )doc"
    );

    // Overload for no arguments (empty initialization)
    m.def(
        "initialize",
        []()
        {
            int argc = 0;
            char** argv = nullptr;
            NeoN::initialize(argc, argv);
        },
        R"doc(
              Initialize NeoN without command line arguments.

              This function initializes the Kokkos execution backend and sets up
              default logging patterns for NeoN using default settings.

              Example:
                  import neon
                  neon.initialize()
          )doc"
    );

#ifdef NF_WITH_MPI_SUPPORT
    m.def(
        "set_default_communicator",
        [](nb::object comm)
        {
            if (comm.is_none())
            {
                NeoN::mpi::setDefaultCommunicator(MPI_COMM_NULL);
                return;
            }
            // an mpi4py communicator, or its Fortran handle (comm.py2f())
            auto handle = nb::hasattr(comm, "py2f") ? nb::cast<int>(comm.attr("py2f")())
                                                    : nb::cast<int>(comm);
            NeoN::mpi::setDefaultCommunicator(MPI_Comm_f2c(static_cast<MPI_Fint>(handle)));
        },
        "comm"_a.none(),
        R"doc(
              Set the communicator NeoN uses wherever it defaults to MPI_COMM_WORLD.

              Call it before building a mesh, e.g. to run NeoN on part of the
              processes. The ranks in it decide which rank logs.

              Parameters:
                  comm: an mpi4py communicator, its Fortran handle
                      (``comm.py2f()``), or None for MPI_COMM_WORLD.
          )doc"
    );

    m.def(
        "default_communicator",
        []() { return static_cast<int>(MPI_Comm_c2f(NeoN::mpi::defaultCommunicator())); },
        R"doc(
              The Fortran handle of NeoN's default communicator
              (``mpi4py.MPI.Comm.f2py(handle)`` gives the communicator).
          )doc"
    );

    m.def(
        "has_default_communicator",
        &NeoN::mpi::hasDefaultCommunicator,
        R"doc(
              Whether a default communicator other than MPI_COMM_WORLD is set.
          )doc"
    );
#endif

    m.def(
        "finalize",
        &NeoN::finalize,
        R"doc(
              Finalize NeoN and clean up resources.

              This function finalizes the Kokkos execution backend and performs
              necessary cleanup. It should be called before the program exits
              or when NeoN is no longer needed.

              Example:
                  import neon
                  neon.initialize()
                  # ... use NeoN functionality ...
                  neon.finalize()
          )doc"
    );
}

} // namespace NeoN::bindings
