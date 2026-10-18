// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "NeoN/linearAlgebra/linearSystem.hpp"
#include "NeoN/linearAlgebra/cooSparsityPattern.hpp"
#include "NeoN/linearAlgebra/csrSparsityPattern.hpp"
#include "NeoN/linearAlgebra/matrix.hpp"
#include "NeoN/linearAlgebra/solver.hpp"
#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace NeoN::bindings
{

template<typename ValueType, typename IndexType>
void declare_matrix(nb::module_& m, const std::string& name)
{
    using Mat = la::Matrix<ValueType, la::CsrSparsityPattern<IndexType>>;

    nb::class_<Mat>(m, name.c_str())
        .def("exec", &Mat::exec, "Get the executor this matrix's data lives on")
        .def("n_rows", &Mat::nRows, "Number of rows")
        .def("n_non_zeros", &Mat::nNonZeros, "Number of non-zero values")
        .def(
            "values",
            static_cast<Vector<ValueType>& (Mat::*)()>(&Mat::values),
            nb::rv_policy::reference_internal,
            "Non-zero values of the CSR matrix (zero-copy reference)"
        )
        .def(
            "col_idxs",
            &Mat::colIdxs,
            nb::rv_policy::reference_internal,
            "Column indices of the CSR matrix (zero-copy reference)"
        )
        .def(
            "row_offs",
            &Mat::rowOffs,
            nb::rv_policy::reference_internal,
            "Row offsets of the CSR matrix (zero-copy reference)"
        );
}

template<typename MatrixValueType, typename RHSValueType = MatrixValueType>
void declare_linear_system(nb::module_& m, const std::string& suffix)
{
    using LS = la::LinearSystem<MatrixValueType, RHSValueType>;
    // SystemMatrixType isn't a re-exported nested type on LS; it defaults to this.
    using Mat = la::CSRMatrix<MatrixValueType, localIdx>;

    nb::class_<LS>(m, ("LinearSystem" + suffix).c_str())
        .def("reset", &LS::reset)
        .def("copy_to_host", &LS::copyToHost)
        .def(
            "matrix",
            static_cast<Mat& (LS::*)()>(&LS::matrix),
            nb::rv_policy::reference_internal,
            "Get the assembled CSR system matrix"
        )
        .def(
            "rhs",
            static_cast<Vector<RHSValueType>& (LS::*)()>(&LS::rhs),
            nb::rv_policy::reference_internal,
            "Get the rhs vector"
        );
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

    declare_matrix<scalar, localIdx>(m, "CSRMatrix");

    declare_linear_system<scalar>(m, "Scalar");
    declare_linear_system<Vec3>(m, "Vector");
    declare_linear_system<scalar, Vec3>(m, "ScalarMatrixVectorRhs");
}

} // namespace NeoN::bindings
