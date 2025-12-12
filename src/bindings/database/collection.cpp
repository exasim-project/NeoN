// SPDX-FileCopyrightText: 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "NeoN/core/database/collection.hpp"
#include "NeoN/core/database/database.hpp"
#include "../bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

// Specialization to indicate Collection is not copy constructible
// Must be defined before nb::class_<Collection> is instantiated
NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)
template<>
struct is_copy_constructible<NeoN::Collection> : std::false_type
{
};
NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)

namespace NeoN::bindings
{

void registerCollection(nb::module_& m)
{
    // Collection class - type-erased interface for collection types
    // Note: Collection has deleted copy constructor, so we don't expose copy semantics
    nb::class_<NeoN::Collection>(m, "Collection", "A type-erased interface for collection types")
        .def(
            "doc",
            [](NeoN::Collection& col, const std::string& id) -> NeoN::Document&
            { return col.doc(id); },
            nb::rv_policy::reference_internal,
            "id"_a,
            "Retrieve a document by its ID"
        )
        .def("size", &NeoN::Collection::size, "Get the number of documents in the collection")
        .def("type", &NeoN::Collection::type, "Get the type of the collection")
        .def("name", &NeoN::Collection::name, "Get the name of the collection")
        .def(
            "db",
            [](NeoN::Collection& col) -> NeoN::Database& { return col.db(); },
            nb::rv_policy::reference_internal,
            "Get the database containing this collection"
        )
        .def(
            "__repr__",
            [](const NeoN::Collection& col)
            { return "<Collection name='" + col.name() + "' type='" + col.type() + "'>"; }
        )
        .def(
            "__str__",
            [](const NeoN::Collection& col)
            { return "Collection(" + col.name() + ", size=" + std::to_string(col.size()) + ")"; }
        )
        .def("__len__", [](const NeoN::Collection& col) { return col.size(); });
}

} // namespace NeoN::bindings
