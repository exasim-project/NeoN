// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include "cell_type.H"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>

#include <array>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

namespace ibm
{

namespace
{

//! `[i, j, k]`, for an error message that names the offending cell (design §10).
std::string cellName(int i, int j, int k)
{
    return "[" + std::to_string(i) + ", " + std::to_string(j) + ", " + std::to_string(k) + "]";
}

//! A `double` with enough digits to be recognisable in a message.
std::string realName(amrex::Real v)
{
    std::ostringstream os;
    os.precision(6);
    os << v;
    return os.str();
}

//! The F10 ghost-width contract, made executable (design §2.1, review F10).
//!
//! Pass 2 reads a face neighbour of every valid cell, and pass 1 evaluates the
//! marker on the grown box, reading the geometry there. Both are requirements,
//! not conveniences; the message shape is the compiled surface's standard (api
//! §8) — the width it needs and the width it has. The geometry half lives in
//! `geometry_view.H` as `requireGeometryGhosts`, because `validateCellType`
//! needs exactly the same check for exactly the same reason (B28-R, I1).
void requireMarkerGhostWidth(const CellTypeFab& ct, const IbmGeometryFab& g)
{
    requireGeometryLayout(g, "classify_default");

    if (ct.boxArray() != g.boxArray())
        throw std::runtime_error(
            "classify_default: the marker and the immersed-body geometry must share a BoxArray "
            "— the marker is a field, not a list, and nothing downstream realigns them"
        );

    const int ctNGrow = ct.nGrowVect().min();
    if (ctNGrow < MARKER_NGROW)
        throw std::runtime_error(
            "classify_default: the marker's second pass reads a face neighbour of every valid "
            "cell, so it needs a ghost width of at least "
            + std::to_string(MARKER_NGROW) + ", but the CellTypeFab has " + std::to_string(ctNGrow)
            + "; grow the marker"
        );

    requireGeometryGhosts("classify_default", g, ctNGrow);
}

} // namespace

void classifyDefault(CellTypeFab& ct, const IbmGeometryFab& g, const amrex::Geometry& geom)
{
    requireMarkerGhostWidth(ct, g);

    // Pass 1 — fluid / non-fluid, on the GROWN box: a WALL cell at a box edge
    // needs its neighbours' markers, and the wall kernel reads them without a
    // second pass.
    for (amrex::MFIter mfi(ct); mfi.isValid(); ++mfi)
    {
        auto const& m = ct.array(mfi);
        const IbmGeometryView gv = makeGeometryView(g, mfi);
        amrex::ParallelFor(
            mfi.growntilebox(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            { m(i, j, k) = (gv.sdf(i, j, k) <= 0.0) ? ibm::SOLID : ibm::FLUID; }
        );
    }
    ct.FillBoundary(geom.periodicity()); // markers across box and periodic edges

    // Pass 2 — a FLUID cell with a SOLID face neighbour is WALL. It runs on the
    // valid box, so at a NON-PERIODIC domain edge the outer ghost keeps pass 1's
    // SOLID/FLUID: the correct fluid/solid state, never a garbage value, and a
    // deliberate asymmetry with the periodic ghost region.
    for (amrex::MFIter mfi(ct); mfi.isValid(); ++mfi)
    {
        auto const& m = ct.array(mfi);
        amrex::ParallelFor(
            mfi.validbox(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                if (m(i, j, k) == ibm::SOLID) return;
                const bool touches = m(i - 1, j, k) == ibm::SOLID || m(i + 1, j, k) == ibm::SOLID
                                  || m(i, j - 1, k) == ibm::SOLID || m(i, j + 1, k) == ibm::SOLID
                                  || m(i, j, k - 1) == ibm::SOLID || m(i, j, k + 1) == ibm::SOLID;
                if (touches) m(i, j, k) = ibm::WALL;
            }
        );
    }
    ct.FillBoundary(geom.periodicity());

    validateCellType(ct, g); // M4 + M5, once per classification (api §8)
}

void validateCellType(const CellTypeFab& ct, const IbmGeometryFab& g)
{
    requireGeometryLayout(g, "validate_cell_type");

    if (ct.boxArray() != g.boxArray())
        throw std::runtime_error(
            "validate_cell_type: the marker and the immersed-body geometry must share a BoxArray"
        );

    // The pass below runs over the MARKER's fab box and reads the GEOMETRY's
    // Array4 at the same indices, so a marker wider than the geometry is an
    // out-of-bounds read — silent garbage in a release build, i.e. a spurious
    // M5 sentence or a segfault (B28-R, I1). `classifyDefault` is safe because
    // `requireMarkerGhostWidth` runs first; this entry point is bound
    // standalone (`blockamr.validate_cell_type`) precisely so the M4/M5 red
    // paths are reachable, so it has to check for itself.
    //
    // Only the geometry-vs-marker half is checked here, not `MARKER_NGROW`:
    // validation of a zero-ghost marker is a legitimate call (its fab box is
    // then its valid box and nothing is read out of bounds), whereas the
    // MARKER_NGROW floor is a requirement of the *classification*'s second
    // pass, which this function does not perform.
    requireGeometryGhosts("validate_cell_type", g, ct.nGrowVect().min());

    // A device lambda cannot throw, so the first bad cell is captured device-side
    // — an atomicCAS claim on slot 4 makes the report singular — and the sentence
    // is built host-side. One pass over the grown box, once per classification.
    amrex::Gpu::DeviceVector<int> badIdx(5);
    amrex::Gpu::DeviceVector<amrex::Real> badSdf(1);
    {
        const std::array<int, 5> zeroIdx {0, 0, 0, 0, 0};
        const amrex::Real zeroSdf = 0.0;
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, zeroIdx.begin(), zeroIdx.end(), badIdx.begin());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, &zeroSdf, &zeroSdf + 1, badSdf.begin());
    }
    int* bi = badIdx.data();
    amrex::Real* bs = badSdf.data();

    for (amrex::MFIter mfi(ct); mfi.isValid(); ++mfi)
    {
        auto const& m = ct.const_array(mfi);
        const IbmGeometryView gv = makeGeometryView(g, mfi);
        amrex::ParallelFor(
            mfi.fabbox(), // ghosts included: M4 covers them
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                const std::uint8_t v = m(i, j, k);
                const amrex::Real s = gv.sdf(i, j, k);
                const bool m4 = (v > ibm::FLUID);
                const bool m5 = (v == ibm::WALL && s <= 0.0) || (v == ibm::SOLID && s > 0.0);
                if (!(m4 || m5)) return;
                if (amrex::Gpu::Atomic::CAS(bi + 4, 0, 1) != 0) return; // first claim wins
                bi[0] = i;
                bi[1] = j;
                bi[2] = k;
                bi[3] = static_cast<int>(v);
                bs[0] = s;
            }
        );
    }
    amrex::Gpu::streamSynchronize();

    std::array<int, 5> hostIdx {0, 0, 0, 0, 0};
    amrex::Real hostSdf = 0.0;
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, badIdx.begin(), badIdx.end(), hostIdx.begin());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, badSdf.begin(), badSdf.end(), &hostSdf);

    if (hostIdx[4] == 0) return;

    const std::string where = cellName(hostIdx[0], hostIdx[1], hostIdx[2]);
    const int v = hostIdx[3];

    if (v > static_cast<int>(FLUID))
        throw std::runtime_error(
            "classify wrote the value " + std::to_string(v) + " at cell " + where
            + ", which is not one of SOLID (0), WALL (1) or FLUID (2): a fourth value falls "
              "into the wall kernel's else branch and is silently treated as FLUID"
        );

    if (v == static_cast<int>(WALL))
        throw std::runtime_error(
            "cell " + where + " is marked WALL but its sdf is " + realName(hostSdf)
            + " <= 0: a WALL cell inside the body has no wall geometry to close against, and "
              "the kernel divides by a distance that is not there"
        );

    throw std::runtime_error(
        "cell " + where + " is marked SOLID but its sdf is " + realName(hostSdf)
        + " > 0: a SOLID cell in the fluid is pinned rather than solved, so the equation "
          "silently loses that cell"
    );
}

void pinSolid(amrex::MultiFab& phi, const CellTypeFab& ct, amrex::Real pin, int ncomp)
{
    if (phi.boxArray() != ct.boxArray())
        throw std::runtime_error(
            "pin_solid: the field and the marker must share a BoxArray — the marker is a field "
            "with the same decomposition as the fields it accompanies"
        );
    if (ncomp < 1 || ncomp > phi.nComp())
        throw std::runtime_error(
            "pin_solid: asked to pin " + std::to_string(ncomp) + " components of a field with "
            + std::to_string(phi.nComp())
        );

    for (amrex::MFIter mfi(phi); mfi.isValid(); ++mfi)
    {
        auto const& p = phi.array(mfi);
        auto const& m = ct.const_array(mfi);
        amrex::ParallelFor(
            mfi.validbox(),
            ncomp,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n)
            {
                if (m(i, j, k) == ibm::SOLID) p(i, j, k, n) = pin;
            }
        );
    }
    amrex::Gpu::streamSynchronize();
}

} // namespace ibm

void registerCellType(nb::module_& m)
{
    using namespace amrex;

    nb::enum_<ibm::CellType>(m, "CellType", nb::is_arithmetic())
        .value("SOLID", ibm::CellType::SOLID)
        .value("WALL", ibm::CellType::WALL)
        .value("FLUID", ibm::CellType::FLUID);

    nb::class_<ibm::CellTypeFab>(m, "CellTypeFab")
        .def(
            "__init__",
            [](ibm::CellTypeFab* self, const BoxArray& ba, const DistributionMapping& dm, int ngrow)
            {
                MFInfo info;
                info.SetAllocSingleChunk(true);
                new (self) ibm::CellTypeFab(ba, dm, 1, ngrow, info);
            },
            nb::arg("ba"),
            nb::arg("dm"),
            nb::arg("ngrow")
        )
        // Test binding (api §4): the ONLY way to put in front of the validation a
        // marker the classification cannot produce — the M4 and M5 red paths.
        .def(
            "set_val",
            [](ibm::CellTypeFab& ct, int val)
            {
                if (val < 0 || val > 255)
                    throw std::runtime_error(
                        "CellTypeFab.set_val: the marker is one byte; " + std::to_string(val)
                        + " does not fit"
                    );
                ct.setVal(static_cast<std::uint8_t>(val));
            },
            nb::arg("val")
        )
        .def("n_grow", [](const ibm::CellTypeFab& ct) { return ct.nGrow(); })
        .def("box_array", [](const ibm::CellTypeFab& ct) { return ct.boxArray(); })
        .def(
            "fill_boundary",
            [](ibm::CellTypeFab& ct, const Geometry& geom) { ct.FillBoundary(geom.periodicity()); },
            nb::arg("geom")
        );

    m.attr("MARKER_NGROW") = ibm::MARKER_NGROW;
    m.attr("IBM_GEOM_NCOMP") = ibm::GEOM_NCOMP;

    m.def(
        "classify_default",
        &ibm::classifyDefault,
        nb::arg("ct"),
        nb::arg("geom_ibm"),
        nb::arg("geom")
    );

    m.def("validate_cell_type", &ibm::validateCellType, nb::arg("ct"), nb::arg("geom_ibm"));

    m.def(
        "pin_solid", &ibm::pinSolid, nb::arg("phi"), nb::arg("ct"), nb::arg("pin"), nb::arg("ncomp")
    );

    // TEST binding (api §4) — read by `test_ibm_cell_type.py` (M4, M5, the WALL
    // predicate, the ghost-fill row) and by nothing on an evaluate path. A free
    // function, underscore-private, so it never enters CellTypeFab's vocabulary.
    // Returns a fresh Fortran-ordered HOST copy owned by a capsule, never a
    // pointer into device memory. `mfi` is an MFIterator over any MultiFab that
    // shares the marker's BoxArray and DistributionMapping.
    m.def(
        "_cell_type_numpy",
        [](ibm::CellTypeFab& ct, nb::object mfiObj, bool grown)
        {
            auto& mfi = nb::cast<MFIter&>(mfiObj.attr("get")());
            auto& fab = ct[mfi];
            const Box bx = grown ? fab.box() : mfi.validbox();
            const int nx = bx.length(0);
            const int ny = bx.length(1);
            const int nz = bx.length(2);
            const std::size_t nelems = static_cast<std::size_t>(nx) * ny * nz;

            BaseFab<std::uint8_t> hostFab(bx, 1, The_Pinned_Arena());
            const bool srcOnDevice =
                ct.arena()->isDeviceAccessible() && !ct.arena()->isHostAccessible();
            if (srcOnDevice)
                hostFab.template copy<RunOn::Device>(
                    fab, bx, SrcComp {0}, DestComp {0}, NumComps {1}
                );
            else
                hostFab.template copy<RunOn::Host>(
                    fab, bx, SrcComp {0}, DestComp {0}, NumComps {1}
                );
            amrex::Gpu::streamSynchronize();

            auto* hostBuf = new std::uint8_t[nelems];
            auto owner = nb::capsule(
                hostBuf, [](void* p) noexcept { delete[] static_cast<std::uint8_t*>(p); }
            );
            std::memcpy(hostBuf, hostFab.dataPtr(), nelems * sizeof(std::uint8_t));

            std::size_t shape[3] = {
                static_cast<std::size_t>(nx),
                static_cast<std::size_t>(ny),
                static_cast<std::size_t>(nz)
            };
            return nb::ndarray<nb::numpy, std::uint8_t, nb::ndim<3>>(
                hostBuf,
                3,
                shape,
                owner,
                nullptr,
                nb::dtype<std::uint8_t>(),
                nb::device::cpu::value,
                0,
                'F'
            );
        },
        nb::arg("ct"),
        nb::arg("mfi"),
        nb::arg("grown") = false
    );
}
