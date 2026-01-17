// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>

#include "NeoN/core/tokenList.hpp"
#include "NeoN/core/dictionary.hpp"
#include "NeoN/core/input.hpp"
#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace NeoN::bindings
{

void registerInputs(nb::module_& m)
{


    // TokenList bindings - simplified version without std::any direct access
    nb::class_<NeoN::TokenList>(m, "TokenList", "A list of tokens storing values of any type")
        .def(nb::init<>(), "Create an empty TokenList")
        .def(nb::init<const NeoN::TokenList&>(), "Copy constructor")
        .def("empty", &NeoN::TokenList::empty, "Check if the token list is empty")
        .def("size", &NeoN::TokenList::size, "Get the size of the token list")
        .def("remove", &NeoN::TokenList::remove, "index"_a, "Remove value at specified index")
        // Add typed get methods for common types
        .def(
            "get_int",
            [](NeoN::TokenList& self, size_t idx) -> int { return self.get<int>(idx); },
            "idx"_a,
            "Get int value at index"
        )
        .def(
            "get_double",
            [](NeoN::TokenList& self, size_t idx) -> double { return self.get<double>(idx); },
            "idx"_a,
            "Get double value at index"
        )
        .def(
            "get_string",
            [](NeoN::TokenList& self, size_t idx) -> std::string
            { return self.get<std::string>(idx); },
            "idx"_a,
            "Get string value at index"
        )
        .def(
            "get_bool",
            [](NeoN::TokenList& self, size_t idx) -> bool { return self.get<bool>(idx); },
            "idx"_a,
            "Get bool value at index"
        )
        // Add typed insert methods
        .def(
            "insert_int",
            [](NeoN::TokenList& self, int value) { self.insert(std::any(value)); },
            "value"_a,
            "Insert int value"
        )
        .def(
            "insert_double",
            [](NeoN::TokenList& self, double value) { self.insert(std::any(value)); },
            "value"_a,
            "Insert double value"
        )
        .def(
            "insert_string",
            [](NeoN::TokenList& self, const std::string& value) { self.insert(std::any(value)); },
            "value"_a,
            "Insert string value"
        )
        .def(
            "insert_bool",
            [](NeoN::TokenList& self, bool value) { self.insert(std::any(value)); },
            "value"_a,
            "Insert bool value"
        )
        .def(
            "__repr__",
            [](const NeoN::TokenList& self)
            { return "<TokenList size=" + std::to_string(self.size()) + ">"; }
        );

    // Dictionary bindings - using the correct interface
    nb::class_<NeoN::Dictionary>(m, "Dictionary", "A dictionary storing key-value pairs")
        .def(nb::init<>(), "Create an empty Dictionary")
        .def(nb::init<const NeoN::Dictionary&>(), "Copy constructor")
        .def("empty", &NeoN::Dictionary::empty, "Check if the dictionary is empty")
        .def(
            "size",
            [](const NeoN::Dictionary& self) { return self.keys().size(); },
            "Get number of keys in dictionary"
        )
        .def("contains", &NeoN::Dictionary::contains, "key"_a, "Check if key exists in dictionary")
        .def("remove", &NeoN::Dictionary::remove, "key"_a, "Remove entry with specified key")
        .def("keys", &NeoN::Dictionary::keys, "Get all keys in the dictionary")
        // Add typed get methods for common types
        .def(
            "get_int",
            [](NeoN::Dictionary& self, const std::string& key) -> int
            { return self.get<int>(key); },
            "key"_a,
            "Get int value for key"
        )
        .def(
            "get_double",
            [](NeoN::Dictionary& self, const std::string& key) -> double
            { return self.get<double>(key); },
            "key"_a,
            "Get double value for key"
        )
        .def(
            "get_string",
            [](NeoN::Dictionary& self, const std::string& key) -> std::string
            { return self.get<std::string>(key); },
            "key"_a,
            "Get string value for key"
        )
        .def(
            "get_bool",
            [](NeoN::Dictionary& self, const std::string& key) -> bool
            { return self.get<bool>(key); },
            "key"_a,
            "Get bool value for key"
        )
        // Add typed insert methods
        .def(
            "insert_int",
            [](NeoN::Dictionary& self, const std::string& key, int value)
            { self.insert(key, std::any(value)); },
            "key"_a,
            "value"_a,
            "Insert int value"
        )
        .def(
            "insert_double",
            [](NeoN::Dictionary& self, const std::string& key, double value)
            { self.insert(key, std::any(value)); },
            "key"_a,
            "value"_a,
            "Insert double value"
        )
        .def(
            "insert_string",
            [](NeoN::Dictionary& self, const std::string& key, const std::string& value)
            { self.insert(key, std::any(value)); },
            "key"_a,
            "value"_a,
            "Insert string value"
        )
        .def(
            "insert_bool",
            [](NeoN::Dictionary& self, const std::string& key, bool value)
            { self.insert(key, std::any(value)); },
            "key"_a,
            "value"_a,
            "Insert bool value"
        )
        .def(
            "__repr__",
            [](const NeoN::Dictionary& self)
            { return "<Dictionary size=" + std::to_string(self.keys().size()) + ">"; }
        );
}

} // namespace NeoN::bindings
