// SPDX-FileCopyrightText: 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "NeoN/core/database/database.hpp"
#include "../bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

// Specialization to indicate Database is not copy constructible
// Must be defined before nb::class_<Database> is instantiated
NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)
template<>
struct is_copy_constructible<NeoN::Database> : std::false_type
{
};
NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)

namespace NeoN::bindings
{

void registerDatabase(nb::module_& m)
{
    // Database class - not copyable, so we need to disable copy
    nb::class_<NeoN::Database>(m, "Database", "A database containing collections")
        .def(nb::init<>(), "Create an empty Database")
        .def(
            "contains",
            &NeoN::Database::contains,
            "name"_a,
            "Check if the database contains a collection with the given name"
        )
        .def("remove", &NeoN::Database::remove, "name"_a, "Remove a collection from the database")
        .def(
            "at",
            [](NeoN::Database& db, const std::string& name) -> NeoN::Collection&
            { return db.at(name); },
            nb::rv_policy::reference_internal,
            "name"_a,
            "Retrieve a collection by its name"
        )
        .def("size", &NeoN::Database::size, "Get the number of collections in the database")
        .def(
            "__repr__",
            [](const NeoN::Database& db)
            { return "<Database size=" + std::to_string(db.size()) + ">"; }
        )
        .def(
            "__str__",
            [](const NeoN::Database& db)
            { return "Database(collections=" + std::to_string(db.size()) + ")"; }
        )
        .def("__len__", [](const NeoN::Database& db) { return db.size(); })
        .def(
            "__contains__",
            [](const NeoN::Database& db, const std::string& name) { return db.contains(name); },
            "name"_a
        )
        .def(
            "__getitem__",
            [](NeoN::Database& db, const std::string& name) -> NeoN::Collection&
            { return db.at(name); },
            nb::rv_policy::reference_internal,
            "name"_a
        );
}

} // namespace NeoN::bindings
