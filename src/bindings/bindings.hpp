// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <nanobind/nanobind.h>

namespace NeoN::bindings
{

// Forward declarations for all binding registration functions
void registerExecutors(nanobind::module_& m);
void registerScalar(nanobind::module_& m);
void registerVec3(nanobind::module_& m);
void registerVectors(nanobind::module_& m);
void registerContainerFreeFunctions(nanobind::module_& m);
void registerBoundaryMesh(nanobind::module_& m);
void registerUnstructuredMesh(nanobind::module_& m);
void registerSurfaceField(nanobind::module_& m);
void registerVolumeField(nanobind::module_& m);
void registerSurfaceInterpolation(nanobind::module_& m);
void registerCoNum(nanobind::module_& m);

// Database bindings
void registerDocument(nanobind::module_& m);
void registerCollection(nanobind::module_& m);
void registerDatabase(nanobind::module_& m);

} // namespace NeoN::bindings
