// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "NeoN/linearAlgebra/linearSystem.hpp"
#include "NeoN/linearAlgebra/sparsityPattern.hpp"
#include "NeoN/linearAlgebra/solver.hpp"
#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace NeoN::bindings
{

template<typename ValueType>
void declare_linear_system(nb::module_& m, const std::string& suffix)
{
    using LS = la::LinearSystem<ValueType, localIdx>;

    nb::class_<LS>(m, ("LinearSystem" + suffix).c_str())
        .def("reset", &LS::reset)
        .def("copy_to_host", &LS::copyToHost);
}

void registerLinearAlgebra(nb::module_& m)
{
    nb::class_<la::SparsityPattern>(m, "SparsityPattern")
        .def("rows", &la::SparsityPattern::rows)
        .def("nnz", &la::SparsityPattern::nnz);

    nb::class_<la::SolverStats>(m, "SolverStats")
        .def_rw("num_iter", &la::SolverStats::numIter)
        .def_rw("initial_residual", &la::SolverStats::initResNorm)
        .def_rw("final_residual", &la::SolverStats::finalResNorm)
        .def_rw("time", &la::SolverStats::solveTime);

    declare_linear_system<scalar>(m, "Scalar");
    declare_linear_system<Vec3>(m, "Vector");
}

} // namespace NeoN::bindings
