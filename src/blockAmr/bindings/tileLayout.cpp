// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// Per-field dispatch metadata for tiled Pallas kernels: one TileLayout per MultiFab (cell or
// face), holding per-tile [offset, sx, sy, sz, box_id] as a device int32 array.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <AMReX.H>
#include <AMReX_MultiFab.H>

#include "bindings.hpp"

namespace nb = nanobind;
using namespace amrex;

// Device tag for The_Device_Arena()-backed tile buffers; CPU on a CPU-only AMReX build so JAX's
// from_dlpack does not request a nonexistent cuda backend.
#if defined(AMREX_USE_CUDA)
constexpr int kTileDevice = nb::device::cuda::value;
#elif defined(AMREX_USE_HIP)
constexpr int kTileDevice = nb::device::rocm::value;
#elif defined(AMREX_USE_SYCL)
constexpr int kTileDevice = nb::device::oneapi::value;
#else
constexpr int kTileDevice = nb::device::cpu::value;
#endif


struct TileLayout
{
    int32_t* d_tiles = nullptr; // device: [n_tiles_padded * 5]
    int n_tiles = 0;
    int n_tiles_padded = 0;
    int n_boxes = 0;
    int n_boxes_padded = 0;
    int bf = 0;
    int ng = 0;

    ~TileLayout()
    {
        if (d_tiles) The_Device_Arena()->free(d_tiles);
    }

    // Non-copyable (owns device memory)
    TileLayout(const TileLayout&) = delete;
    TileLayout& operator=(const TileLayout&) = delete;
    TileLayout(TileLayout&& o) noexcept
        : d_tiles(o.d_tiles), n_tiles(o.n_tiles), n_tiles_padded(o.n_tiles_padded),
          n_boxes(o.n_boxes), n_boxes_padded(o.n_boxes_padded), bf(o.bf), ng(o.ng)
    {
        o.d_tiles = nullptr;
    }
    TileLayout& operator=(TileLayout&& o) noexcept
    {
        if (this != &o)
        {
            if (d_tiles) The_Device_Arena()->free(d_tiles);
            d_tiles = o.d_tiles;
            n_tiles = o.n_tiles;
            n_tiles_padded = o.n_tiles_padded;
            n_boxes = o.n_boxes;
            n_boxes_padded = o.n_boxes_padded;
            bf = o.bf;
            ng = o.ng;
            o.d_tiles = nullptr;
        }
        return *this;
    }
    TileLayout() = default;
};


static TileLayout buildTileLayout(MultiFab& mf, int bf)
{
    if (!mf.singleChunkPtr())
        throw std::runtime_error("build_tile_layout: MultiFab not in single-chunk mode");

    int ng = mf.nGrow();
    int nc = mf.nComp();
    constexpr int FIELDS = 5;

    int n_boxes = 0;
    size_t n_tiles = 0;
    for (MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const auto& bx = mf[mfi].box();
        int vNx = bx.length(0) - 2 * ng;
        int vNy = bx.length(1) - 2 * ng;
        int vNz = bx.length(2) - 2 * ng;
        n_tiles += (size_t)(vNx / bf) * (vNy / bf) * (vNz / bf);
        n_boxes++;
    }

    size_t n_tiles_padded = 1;
    while (n_tiles_padded < n_tiles)
        n_tiles_padded <<= 1;
    int n_boxes_padded = 1;
    while (n_boxes_padded < n_boxes)
        n_boxes_padded <<= 1;

    auto* packed = new int32_t[n_tiles_padded * FIELDS];

    size_t t = 0;
    size_t box_offset = 0;
    int box_id = 0;
    for (MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const auto& bx = mf[mfi].box();
        int Nx = bx.length(0);
        int Ny = bx.length(1);
        int Nz = bx.length(2);
        int vNx = Nx - 2 * ng;
        int vNy = Ny - 2 * ng;
        int vNz = Nz - 2 * ng;

        if (box_offset > (size_t)INT32_MAX)
            throw std::runtime_error("build_tile_layout: buffer offset exceeds int32 range");

        int32_t sx = 1;
        int32_t sy = Nx;
        int32_t sz = Nx * Ny;

        for (int ti = 0; ti < vNx / bf; ++ti)
        {
            for (int tj = 0; tj < vNy / bf; ++tj)
            {
                for (int tk = 0; tk < vNz / bf; ++tk)
                {
                    int ci = ng + ti * bf;
                    int cj = ng + tj * bf;
                    int ck = ng + tk * bf;
                    int32_t offset = (int32_t)box_offset + ci * sx + cj * sy + ck * sz;

                    size_t base = t * FIELDS;
                    packed[base + 0] = offset;
                    packed[base + 1] = sx;
                    packed[base + 2] = sy;
                    packed[base + 3] = sz;
                    packed[base + 4] = box_id;
                    ++t;
                }
            }
        }

        box_offset += (size_t)Nx * Ny * Nz * nc;
        ++box_id;
    }

    // Pad with copies of tile 0
    for (size_t p = t; p < n_tiles_padded; ++p)
    {
        size_t dst = p * FIELDS;
        for (int f = 0; f < FIELDS; ++f)
            packed[dst + f] = packed[f];
    }

    size_t nbytes = n_tiles_padded * FIELDS * sizeof(int32_t);
    auto* dev_ptr = static_cast<int32_t*>(The_Device_Arena()->alloc(nbytes));
    Gpu::htod_memcpy(dev_ptr, packed, nbytes);
    delete[] packed;

    TileLayout layout;
    layout.d_tiles = dev_ptr;
    layout.n_tiles = (int)n_tiles;
    layout.n_tiles_padded = (int)n_tiles_padded;
    layout.n_boxes = n_boxes;
    layout.n_boxes_padded = n_boxes_padded;
    layout.bf = bf;
    layout.ng = ng;
    return layout;
}


void registerTileLayout(nb::module_& m)
{
    nb::class_<TileLayout>(m, "TileLayout")
        .def_prop_ro(
            "tiles",
            [](TileLayout& t)
            {
                size_t shape[1] = {(size_t)t.n_tiles_padded * 5};
                return nb::ndarray<nb::jax, int32_t, nb::ndim<1>>(
                    t.d_tiles, 1, shape, nb::handle(), nullptr, nb::dtype<int32_t>(), kTileDevice, 0
                );
            },
            nb::rv_policy::reference_internal
        )
        .def_ro("n_tiles", &TileLayout::n_tiles)
        .def_ro("n_tiles_padded", &TileLayout::n_tiles_padded)
        .def_ro("n_boxes", &TileLayout::n_boxes)
        .def_ro("n_boxes_padded", &TileLayout::n_boxes_padded)
        .def_ro("bf", &TileLayout::bf)
        .def_ro("ng", &TileLayout::ng);

    m.def(
        "build_tile_layout",
        [](MultiFab& mf, int bf) { return buildTileLayout(mf, bf); },
        nb::arg("mf"),
        nb::arg("bf") = 8,
        nb::rv_policy::move,
        "Build TileLayout for a MultiFab. Works on cell or face MultiFabs."
    );
}
