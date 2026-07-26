// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <AMReX_IndexType.H>

namespace nb = nanobind;

void registerIndexType(nb::module_& m)
{
    using namespace amrex;

    auto index_type = nb::class_<IndexType>(m, "IndexType");

    nb::enum_<IndexType::CellIndex>(index_type, "CellIndex")
        .value("CELL", IndexType::CellIndex::CELL)
        .value("NODE", IndexType::CellIndex::NODE);

    index_type.def(nb::init<>())
        .def(
            "__init__",
            [](IndexType* self,
               IndexType::CellIndex i,
               IndexType::CellIndex j,
               IndexType::CellIndex k) { new (self) IndexType(i, j, k); },
            nb::arg("i"),
            nb::arg("j"),
            nb::arg("k")
        )
        .def(
            "__repr__",
            [](const IndexType& t)
            {
                std::string s = "<blockamr.IndexType (";
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    s += t.cellCentered(d) ? "C" : "N";
                    if (d < AMREX_SPACEDIM - 1) s += ",";
                }
                return s + ")>";
            }
        )
        .def("__getitem__", [](const IndexType& t, int i) { return t[i]; })
        .def("__eq__", [](const IndexType& a, const IndexType& b) { return a == b; })
        .def("cell_centered", [](const IndexType& t) { return t.cellCentered(); })
        .def(
            "cell_centered",
            [](const IndexType& t, int dir) { return t.cellCentered(dir); },
            nb::arg("dir")
        )
        .def("node_centered", [](const IndexType& t) { return t.nodeCentered(); })
        .def(
            "node_centered",
            [](const IndexType& t, int dir) { return t.nodeCentered(dir); },
            nb::arg("dir")
        )
        .def("set", &IndexType::set, nb::arg("dir"))
        .def("unset", &IndexType::unset, nb::arg("dir"))
        .def("test", &IndexType::test, nb::arg("dir"))
        .def("to_int_vect", &IndexType::toIntVect)
        .def_static("cell_type", &IndexType::TheCellType)
        .def_static("node_type", &IndexType::TheNodeType);
}
