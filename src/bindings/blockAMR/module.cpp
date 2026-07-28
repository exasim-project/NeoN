// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>

#include "bindings.hpp"

namespace nb = nanobind;

NB_MODULE(_blockamr, m)
{
    m.doc() = "blockAMR: nanobind Python bindings for AMReX";

    registerInit(m);
    registerIndexType(m);
    registerBox(m);
    registerMultiFab(m);
    registerGeometry(m);
    registerPlotfile(m);
    registerTagBox(m);
    registerFillPatch(m);
    registerAmrCore(m);
    registerLinOp(m);
    registerStencilKernels(m);
    registerCellType(m);
    registerGhostCell(m);
    registerTileLayout(m);
    registerWallTable(m);
    registerBandTable(m);
}
