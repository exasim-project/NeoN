// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include "NeoN/core/primitives/scalar.hpp"
#include "NeoN/core/primitives/vec3.hpp"
#include "NeoN/core/vector/vector.hpp"

namespace NeoN::io
{

/** Geometric quantities derived from mesh topology and point coordinates. */
struct MeshGeometry
{
    Vector<scalar> cellVolumes;
    Vector<Vec3> cellCenters;
    Vector<Vec3> faceAreas;
    Vector<Vec3> faceCenters;
    Vector<scalar> magFaceAreas;
};

} // namespace NeoN::io
