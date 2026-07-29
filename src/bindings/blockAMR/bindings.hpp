// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

void registerInit(nb::module_& m);
void registerIndexType(nb::module_& m);
void registerBox(nb::module_& m);
void registerMultiFab(nb::module_& m);
void registerGeometry(nb::module_& m);
void registerPlotfile(nb::module_& m);
void registerAmrCore(nb::module_& m);
void registerTagBox(nb::module_& m);
void registerFillPatch(nb::module_& m);
void registerLinOp(nb::module_& m);
void registerStencilKernels(nb::module_& m);
void registerCellType(nb::module_& m);
void registerGhostCell(nb::module_& m);
void registerStl(nb::module_& m);
void registerWallFrame(nb::module_& m);
void registerRobinClosure(nb::module_& m);
void registerLaplacianGhostCell(nb::module_& m);
void registerDivGhostCell(nb::module_& m);
void registerGradGhostCell(nb::module_& m);
void registerSourceGhostCell(nb::module_& m);
void registerTileLayout(nb::module_& m);
