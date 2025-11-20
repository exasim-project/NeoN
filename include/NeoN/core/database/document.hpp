// SPDX-FileCopyrightText: 2024 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <functional>
#include <atomic>

#include "NeoN/core/dictionary.hpp"

namespace NeoN
{

/**
 * @typedef DocumentValidator
 * @brief A type alias for a function that validates a Dictionary object.
 *
 * This type alias represents a function that takes a Dictionary object as an argument
 * and returns a boolean value indicating whether the Dictionary is valid or not.
 *
 * Example usage:
 * @code
 * DocumentValidator validator = [](Dictionary dict) -> bool {
 *     // Validation logic here
 *     return true; // or false based on validation
 * };
 * @endcode
 */
using DocumentValidator = std::function<bool(Dictionary)>;

/**
 * @brief Checks if a Dictionary object has an "id" key.
 *
 * This function checks if the given Dictionary object has an "id" key.
 *
 * @param doc The Dictionary object to check.
 * @return true if the Dictionary has an "id" key, false otherwise.
 */
bool hasId(Dictionary doc);


/**
 * @class Document
 * @brief A class representing a document in a database.
 *
 * The Document class represents a document in a database. It is a subclass of the Dictionary
 * class and provides additional functionality for validating the document and retrieving its ID.
 */
class Document : public Dictionary
{
public:

    /**
     * @brief Constructs a Document with a unique ID.
     */
    Document();

    /**
     * @brief Constructs a Document with the given Dictionary and validator.
     *
     * @param dict The Dictionary object to construct the Document from.
     * @param validator The validator function to use for validating the Document.
     */
    Document(const Dictionary& dict, DocumentValidator validator = hasId);

    /**
     * @brief Validates the Document.
     *
     * This function validates the Document by calling the validator function with the Document's
     * Dictionary object as an argument.
     *
     * @return true if the Document is valid, false otherwise.
     */
    bool validate() const;

    /**
     * @brief Retrieves the ID of the Document.
     *
     * @return std::string The ID of the Document.
     */
    std::string id() const { return getRef<std::string>("id"); }

private:

    /**
     * @brief Generates a unique ID for a Document.
     *
     * @return std::string A unique ID.
     */
    static std::string generateID()
    {
        static std::atomic<int> counter {0};
        return "doc_" + std::to_string(counter++);
    }
    std::string id_;              ///< The ID of the Document.
    DocumentValidator validator_; ///< The validator function for the Document.
};

/**
 * @brief Retrieves the name of a Document.
 *
 * This function retrieves the name of the Document by looking up the "name" key in the Document's
 * Dictionary object.
 *
 * @param doc The Document to retrieve the name from.
 * @return std::string The name of the Document.
 */
const std::string& name(const NeoN::Document& doc);

/**
 * @brief Retrieves the name of a Document.
 *
 * This function retrieves the name of the Document by looking up the "name" key in the Document's
 * Dictionary object.
 *
 * @param doc The Document to retrieve the name from.
 * @return std::string The name of the Document.
 */
std::string& name(NeoN::Document& doc);

} // namespace NeoN
