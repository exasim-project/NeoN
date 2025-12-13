#pragma once

#include "NeoN/timeIntegration/ddt/DdtScheme.hpp"

namespace NeoN
{
namespace timeIntegration
{
namespace ddt
{

/**
 * @brief Backward Euler time-derivative discretisation.
 *
 * Corresponds to OpenFOAM:
 *   ddtSchemes { default Euler; }
 */
class Euler final : public DdtScheme
{
public:
    Euler() = default;
    ~Euler() override = default;

    int nSteps() const override
    {
        return 1;
    }
};

} // namespace ddt
} // namespace timeIntegration
} // namespace NeoN

