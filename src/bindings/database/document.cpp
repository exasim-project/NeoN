// SPDX-FileCopyrightText: 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "NeoN/core/database/document.hpp"
#include "../bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace NeoN::bindings
{

void registerDocument(nb::module_& m)
{
    // Document class
    nb::class_<NeoN::Document>(m, "Document", "A document in a database collection")
        .def(nb::init<>(), "Create a Document with a unique auto-generated ID")
        .def("validate", &NeoN::Document::validate, "Validate the document")
        .def("id", &NeoN::Document::id, "Get the unique ID of the document")
        .def(
            "__repr__", [](const NeoN::Document& doc) { return "<Document id='" + doc.id() + "'>"; }
        )
        .def("__str__", [](const NeoN::Document& doc) { return "Document(" + doc.id() + ")"; });

    // Free function to get document name
    m.def(
        "document_name",
        [](const NeoN::Document& doc) -> const std::string& { return NeoN::name(doc); },
        nb::rv_policy::reference,
        "doc"_a,
        "Get the name of a document"
    );
}

} // namespace NeoN::bindings
