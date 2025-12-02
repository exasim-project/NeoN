// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <numeric>
#include <iostream> // for operator<<, basic_ostream, endl, cerr, ostream

#include "NeoN/core/dictionary.hpp"
#include "NeoN/core/error.hpp"

namespace NeoN
{

/* @brief basic function to convert std::any to std::string
 *
 * The function tries to convert a given std::any to either an
 * int, float, or Dictionary. In case of a dictionary toString is
 * applied recursively
 *
 * @return the resulting string
 */
std::string toString(std::any& in)
{
    try
    {
        return std::to_string(std::any_cast<int>(in));
    }
    catch (const std::bad_any_cast& e)
    {}
    try
    {
        return std::any_cast<std::string>(in);
    }
    catch (const std::bad_any_cast& e)
    {}
    try
    {
        return std::to_string(std::any_cast<float>(in));
    }
    catch (const std::bad_any_cast& e)
    {}
    try
    {
        auto d = std::any_cast<Dictionary>(in);
        std::string ret = "{\n";
        for (auto [k, v] : d.getMap())
        {
            ret += k + ": ";
            ret += toString(v);
            ret += "\n";
        }
        ret += "}";
        return ret;
    }
    catch (const std::bad_any_cast& e)
    {}

    return "???";
}

void logOutRange(
    const std::out_of_range& e,
    const std::string& key,
    const std::unordered_map<std::string, std::any>& data
)
{
    // TODO use NeoN error here
    std::cerr << "Key not found: " << key << " \n"
              << "available keys are: \n"
              << std::accumulate(
                     data.begin(),
                     data.end(),
                     std::string(""),
                     [](const std::string& a, const std::pair<std::string, std::any>& b)
                     { return a + "  - " + b.first + " \n"; }
                 )
              << e.what() << std::endl;
}

Dictionary::Dictionary(const std::unordered_map<std::string, std::any>& keyValuePairs)
    : data_(keyValuePairs)
{}

Dictionary::Dictionary(const std::initializer_list<std::pair<std::string, std::any>>& initList)
{
    for (const auto& pair : initList)
    {
        data_.insert(pair);
    }
}

void Dictionary::insert(const std::string& key, const std::any& value) { data_[key] = value; }

void Dictionary::remove(const std::string& key) { data_.erase(key); }

bool Dictionary::contains(const std::string& key) const { return data_.find(key) != data_.end(); }

std::any& Dictionary::operator[](const std::string& key)
{
    try
    {
        return data_.at(key);
    }
    catch (const std::out_of_range& e)
    {
        logOutRange(e, key, data_);
        throw e;
    }
}

const std::any& Dictionary::operator[](const std::string& key) const
{
    try
    {
        return data_.at(key);
    }
    catch (const std::out_of_range& e)
    {
        logOutRange(e, key, data_);
        throw e;
    }
}

Dictionary& Dictionary::subDict(const std::string& key)
{
    return std::any_cast<Dictionary&>(operator[](key));
}

const Dictionary& Dictionary::subDict(const std::string& key) const
{
    return std::any_cast<const Dictionary&>(operator[](key));
}

bool Dictionary::isDict(const std::string& key) const { return isType<Dictionary>(key); }

// get keys of the dictionary
std::vector<std::string> Dictionary::keys() const
{
    std::vector<std::string> keys;
    for (const auto& pair : data_)
    {
        keys.push_back(pair.first);
    }
    return keys;
}

std::unordered_map<std::string, std::any>& Dictionary::getMap() { return data_; }

const std::unordered_map<std::string, std::any>& Dictionary::getMap() const { return data_; }

std::ostream& operator<<(std::ostream& os, const Dictionary& in)
{
    os << "{\n";
    for (auto [k, v] : in.getMap())
    {
        os << k << ": " << toString(v) << " ";
        os << "\n";
    }
    os << "}" << std::flush;
    return os;
}
} // namespace NeoN
