// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/vector/vector.hpp"
#include "NeoN/mesh/unstructured/unstructuredMesh.hpp"

namespace NeoN::turbulenceModels::DES
{

class maxDeltaxyz
{
public:

    explicit maxDeltaxyz(const UnstructuredMesh& mesh);

    void update();

    const Vector<scalar>& delta() const;

private:

    const UnstructuredMesh& mesh_;
    Vector<scalar> delta_;
};

} // namespace NeoN::turbulenceModels::DES
