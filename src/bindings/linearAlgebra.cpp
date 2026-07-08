// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "NeoN/linearAlgebra/linearSystem.hpp"
#include "NeoN/linearAlgebra/cooSparsityPattern.hpp"
#include "NeoN/linearAlgebra/csrSparsityPattern.hpp"
#include "NeoN/linearAlgebra/solver.hpp"
#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace NeoN::bindings
{

template<typename ValueType>
void declare_linear_system(nb::module_& m, const std::string& suffix)
{
    using LS = la::LinearSystem<ValueType>;

    nb::class_<LS>(m, ("LinearSystem" + suffix).c_str())
        .def("reset", &LS::reset)
        .def("copy_to_host", &LS::copyToHost);
}

void registerLinearAlgebra(nb::module_& m)
{
    nb::class_<la::CsrSparsityPattern<localIdx>>(m, "CsrSparsityPattern")
        .def("rows", &la::CsrSparsityPattern<localIdx>::rows)
        .def("nnz", &la::CsrSparsityPattern<localIdx>::nnz);

    nb::class_<la::CooSparsityPattern<localIdx>>(m, "CooSparsityPattern")
        .def("rows", &la::CooSparsityPattern<localIdx>::rows)
        .def("nnz", &la::CooSparsityPattern<localIdx>::nnz);

    nb::class_<la::SolverStats>(m, "SolverStats").def_rw("entries", &la::SolverStats::entries);

    nb::class_<la::SolverStatsEntry>(m, "SolverStatsEntry")
        .def_rw("num_iter", &la::SolverStatsEntry::numIter)
        .def_rw("initial_residual", &la::SolverStatsEntry::initResNorm)
        .def_rw("final_residual", &la::SolverStatsEntry::finalResNorm)
        .def_rw("time", &la::SolverStatsEntry::solveTime);


    declare_linear_system<scalar>(m, "Scalar");
    declare_linear_system<Vec3>(m, "Vector");
}

} // namespace NeoN::bindings
