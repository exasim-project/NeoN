// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

//! An STL triangulation as an immersed body's signed distance function.
//!
//! One concern: turn a triangulated surface on disk into signed distance
//! *values on a lattice*. AMReX's `STLtools` already owns the hard parts — the
//! ASCII/binary reader, the BVH and the GPU-capable exact point-to-triangle
//! distance — and this file is the thinnest wrapper over the two calls the IBM
//! geometry pipeline needs.
//!
//! **Why a lattice block and not a MultiFab.** `fillSignedDistance` fills a
//! `MultiFab` on a `Geometry`, and the obvious binding would hand it the
//! level's own fab. The pipeline does not consume the sdf that way: every
//! consumer of `mesh.bodies` (`ibm/classify.py`, `ibm/geometry.py`) is pure
//! numpy over an *index range*, and it asks the body for values at points that
//! are the level's cell centres, the same centres grown by up to `MAX_DEPTH`
//! ghosts and — in `_check_resolvable_gap` — the same centres shifted half a
//! cell onto a face. Those are all regular axis-aligned lattices, and none of
//! them is a level MultiFab. So the binding takes the lattice directly:
//! `origin`, `dx` and a cell count, with sample point
//! `origin + (i + 1/2) * dx`. A single-box, rank-local `MultiFab` is built to
//! carry it, because that is the only shape `fillSignedDistance` accepts.
//!
//! The sign is AMReX's and it is already the pipeline's: `fillSignedDistance`
//! is `fill(..., outside = +1, inside = -1)` scaled by the distance, so the
//! value is **positive in the fluid, negative inside the solid** — the
//! convention `ibm/body.py` documents for every analytic body.

#include "../bindings.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>

#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_EB_STL_utils.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_RealBox.H>
#include <AMReX_Vector.H>

#include <array>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

namespace ibm
{

namespace
{

//! A triangulation read once, queried many times.
class StlSurface
{
public:

    StlSurface(
        const std::string& path,
        amrex::Real scale,
        const std::array<amrex::Real, 3>& center,
        bool reverseNormal
    )
    {
        // `STLtools::read_stl_file` calls `amrex::Abort` on a file it cannot
        // open, which takes the process down instead of raising in Python.
        // Refuse first, on every rank, so the caller gets an exception.
        std::ifstream is(path, std::ios::in | std::ios::binary);
        if (!is.good())
            throw std::runtime_error("StlSurface: cannot open STL file '" + path + "' for reading");
        is.close();

        m_stl.read_stl_file(path, scale, {center[0], center[1], center[2]}, reverseNormal ? 1 : 0);
    }

    //! Signed distance at `origin + (i + 1/2) * dx`, `i` in `[0, n)` per axis.
    //!
    //! Returned as an owning Fortran-ordered `(n0, n1, n2)` host array, which
    //! is what the numpy geometry core indexes into.
    nb::ndarray<nb::numpy, amrex::Real, nb::ndim<3>> signedDistanceBlock(
        const std::array<amrex::Real, 3>& origin,
        const std::array<amrex::Real, 3>& dx,
        const std::array<int, 3>& n
    ) const
    {
        using namespace amrex;

        for (int d = 0; d < 3; ++d)
        {
            if (n[d] < 1)
                throw std::runtime_error(
                    "StlSurface.signed_distance_block: the sample count must be at least 1 on "
                    "every axis, got "
                    + std::to_string(n[d]) + " on axis " + std::to_string(d)
                );
            if (!(dx[d] > 0.0))
                throw std::runtime_error(
                    "StlSurface.signed_distance_block: the lattice spacing must be positive on "
                    "every axis, got "
                    + std::to_string(dx[d]) + " on axis " + std::to_string(d)
                );
        }

        const Box domain(IntVect(0, 0, 0), IntVect(n[0] - 1, n[1] - 1, n[2] - 1));
        const RealBox rb(
            {origin[0], origin[1], origin[2]},
            {origin[0] + n[0] * dx[0], origin[1] + n[1] * dx[1], origin[2] + n[2] * dx[2]}
        );
        const Geometry geom(domain, rb, 0, {0, 0, 0});

        // Rank-local by construction: the numpy geometry core already runs
        // per LOCAL box, so this block belongs to the calling rank alone and
        // the fill must not be a collective.
        const BoxArray ba(domain);
        const DistributionMapping dm(Vector<int> {ParallelDescriptor::MyProc()});
        MultiFab mf(ba, dm, 1, 0);

        m_stl.fillSignedDistance(mf, IntVect(0), geom);

        const std::size_t total = static_cast<std::size_t>(n[0]) * static_cast<std::size_t>(n[1])
                                * static_cast<std::size_t>(n[2]);
        auto* data = new Real[total];
        for (MFIter mfi(mf); mfi.isValid(); ++mfi)
            Gpu::dtoh_memcpy(data, mf[mfi].dataPtr(), total * sizeof(Real));

        nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<Real*>(p); });
        std::size_t shape[3] = {
            static_cast<std::size_t>(n[0]),
            static_cast<std::size_t>(n[1]),
            static_cast<std::size_t>(n[2])
        };
        return nb::ndarray<nb::numpy, Real, nb::ndim<3>>(
            data, 3, shape, owner, nullptr, nb::dtype<Real>(), nb::device::cpu::value, 0, 'F'
        );
    }

private:

    amrex::STLtools m_stl;
};

} // namespace

} // namespace ibm

void registerStl(nb::module_& m)
{
    nb::class_<ibm::StlSurface>(m, "StlSurface")
        .def(
            nb::init<const std::string&, amrex::Real, const std::array<amrex::Real, 3>&, bool>(),
            nb::arg("path"),
            nb::arg("scale"),
            nb::arg("center"),
            nb::arg("reverse_normal"),
            "Read an STL triangulation. Every vertex becomes `v * scale + center`; "
            "`reverse_normal` flips the facet winding, and with it which side of the "
            "surface is solid."
        )
        .def(
            "signed_distance_block",
            &ibm::StlSurface::signedDistanceBlock,
            nb::arg("origin"),
            nb::arg("dx"),
            nb::arg("n"),
            "Exact signed distance at the lattice `origin + (i + 1/2) * dx`, positive "
            "outside the surface. Returns an `(n0, n1, n2)` numpy array."
        );
}
