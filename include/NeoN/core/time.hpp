// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <string>

#include "primitives/scalar.hpp"

const std::string NL = "\n";

namespace NeoN
{

using word = std::string;

class ArgList
{

public:

    ArgList([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {};

    [[nodiscard]] bool checkRootCase() const { return true; };
};

class Time
{
public:

    const static word CONTROL_DICT_NAME;

    Time() : time_(0.) {};

    Time(const word, const ArgList) : time_(0.) {};

    [[nodiscard]] word timeName() { return std::to_string(time_); }

    [[nodiscard]] bool loop()
    {
        time_ += 1.;
        return 10.0 >= time_;
    };

    std::ostream& printExecutionTime(std::ostream& os) const { return os; };

private:

    scalar time_;
};

} // namespace NeoN
