// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoN authors

#include "NeoN/core/tokenList.hpp"

namespace NeoN
{
void logOutRange(
    const std::out_of_range& e, const std::size_t& key, const std::vector<std::any>& data
)
{
    std::cerr << "Index is out of bounds: " << key << " \n"
              << "the size is :" << data.size() << " \n"
              << e.what() << std::endl;
}

TokenList::TokenList() : data_(), nextIndex_(0) {}

TokenList::TokenList(const TokenList& other) : data_(other.data_), nextIndex_(other.nextIndex_) {}

TokenList::TokenList(const std::vector<std::any>& data, size_t nextIndex)
    : data_(data), nextIndex_(nextIndex)
{}

TokenList::TokenList(const std::initializer_list<std::any>& initList)
    : data_(initList), nextIndex_(0)
{}

void TokenList::insert(const std::any& value) { data_.push_back(value); }

void TokenList::remove(size_t index)
{
    data_.erase(data_.begin() + static_cast<std::vector<double>::difference_type>(index));
}

[[nodiscard]] bool TokenList::empty() const { return data_.empty(); }

[[nodiscard]] size_t TokenList::size() const { return data_.size(); }

[[nodiscard]] std::vector<std::any>& TokenList::tokens() { return data_; }

} // namespace NeoN
