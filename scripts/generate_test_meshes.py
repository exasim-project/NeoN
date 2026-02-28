#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Generate reference CGNS meshes for NeoN IO tests."""

import gmsh
import pathlib

OUTDIR = pathlib.Path(__file__).resolve().parent.parent / "test/mesh/unstructured/io/meshFiles"


def generate_single_tet():
    gmsh.initialize()
    gmsh.model.add("singleTet")
    geo = gmsh.model.geo
    lc = 10.0  # large mesh size to prevent refinement
    p0 = geo.addPoint(0, 0, 0, lc)
    p1 = geo.addPoint(1, 0, 0, lc)
    p2 = geo.addPoint(0, 1, 0, lc)
    p3 = geo.addPoint(0, 0, 1, lc)
    # 6 edges
    l01 = geo.addLine(p0, p1)
    l12 = geo.addLine(p1, p2)
    l20 = geo.addLine(p2, p0)
    l03 = geo.addLine(p0, p3)
    l13 = geo.addLine(p1, p3)
    l23 = geo.addLine(p2, p3)
    # 4 triangular faces
    cl0 = geo.addCurveLoop([l01, l12, l20])
    s0 = geo.addPlaneSurface([cl0])
    cl1 = geo.addCurveLoop([l01, l13, -l03])
    s1 = geo.addPlaneSurface([cl1])
    cl2 = geo.addCurveLoop([l12, l23, -l13])
    s2 = geo.addPlaneSurface([cl2])
    cl3 = geo.addCurveLoop([l20, l03, -l23])
    s3 = geo.addPlaneSurface([cl3])
    # Volume
    sl = geo.addSurfaceLoop([s0, s1, s2, s3])
    geo.addVolume([sl])
    geo.synchronize()
    # Prevent any mesh refinement
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 10.0)
    gmsh.option.setNumber("Mesh.Optimize", 0)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
    # Ensure 1 element per edge
    for c in range(1, 7):
        gmsh.model.mesh.setTransfiniteCurve(c, 2)  # 2 nodes = 1 element
    gmsh.model.mesh.generate(3)
    gmsh.write(str(OUTDIR / "singleTet.cgns"))
    gmsh.finalize()


def generate_cube_3d():
    gmsh.initialize()
    gmsh.model.add("cube3D")
    box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    surfaces = gmsh.model.getEntities(2)
    names = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]
    for surf, name in zip(surfaces, names):
        tag = gmsh.model.addPhysicalGroup(2, [surf[1]])
        gmsh.model.setPhysicalName(2, tag, name)
    vol_tag = gmsh.model.addPhysicalGroup(3, [box])
    gmsh.model.setPhysicalName(3, vol_tag, "fluid")
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.5)
    gmsh.model.mesh.generate(3)
    gmsh.write(str(OUTDIR / "cube3D.cgns"))
    gmsh.finalize()


def generate_cavity_2d():
    gmsh.initialize()
    gmsh.model.add("cavity2D")
    rect = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
    gmsh.model.occ.synchronize()
    edges = gmsh.model.getEntities(1)
    for dim, tag in edges:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        if abs(com[1]) < 1e-6:
            name = "bottom"
        elif abs(com[1] - 1.0) < 1e-6:
            name = "top"
        elif abs(com[0]) < 1e-6:
            name = "left"
        else:
            name = "right"
        ptag = gmsh.model.addPhysicalGroup(1, [tag])
        gmsh.model.setPhysicalName(1, ptag, name)
    stag = gmsh.model.addPhysicalGroup(2, [rect])
    gmsh.model.setPhysicalName(2, stag, "fluid")
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.25)
    gmsh.model.mesh.generate(2)
    gmsh.write(str(OUTDIR / "cavity2D.cgns"))
    gmsh.finalize()


def generate_mixed_cells():
    """Generate a unit cube with mixed hex+tet+pyramid elements.

    Creates two adjacent boxes: left half is structured hex, right half
    is unstructured tet. Pyramids form at the interface.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("mixedCells")

    # Two adjacent boxes: left hex, right tet
    box1 = gmsh.model.occ.addBox(0, 0, 0, 0.5, 1, 1)
    box2 = gmsh.model.occ.addBox(0.5, 0, 0, 0.5, 1, 1)
    gmsh.model.occ.fragment([(3, box1)], [(3, box2)])
    gmsh.model.occ.synchronize()

    # Add physical groups for boundary surfaces
    surfaces = gmsh.model.getEntities(2)
    for dim, stag in surfaces:
        com = gmsh.model.occ.getCenterOfMass(dim, stag)
        if abs(com[0]) < 1e-6:
            name = "xmin"
        elif abs(com[0] - 1.0) < 1e-6:
            name = "xmax"
        elif abs(com[1]) < 1e-6:
            name = "ymin"
        elif abs(com[1] - 1.0) < 1e-6:
            name = "ymax"
        elif abs(com[2]) < 1e-6:
            name = "zmin"
        elif abs(com[2] - 1.0) < 1e-6:
            name = "zmax"
        else:
            continue  # internal interface
        ptag = gmsh.model.addPhysicalGroup(2, [stag])
        gmsh.model.setPhysicalName(2, ptag, name)

    volumes = gmsh.model.getEntities(3)
    vol_tags = [v[1] for v in volumes]
    vol_tag = gmsh.model.addPhysicalGroup(3, vol_tags)
    gmsh.model.setPhysicalName(3, vol_tag, "fluid")

    # Make the left box (x < 0.3) structured hex
    for dim, vtag in volumes:
        com = gmsh.model.occ.getCenterOfMass(dim, vtag)
        if com[0] < 0.3:
            gmsh.model.mesh.setTransfiniteVolume(vtag)
            bnd = gmsh.model.getBoundary([(dim, vtag)])
            for _, btag in bnd:
                gmsh.model.mesh.setTransfiniteSurface(abs(btag))
                gmsh.model.mesh.setRecombine(2, abs(btag))
                curves = gmsh.model.getBoundary([(2, abs(btag))])
                for _, ctag in curves:
                    gmsh.model.mesh.setTransfiniteCurve(abs(ctag), 4)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.4)
    gmsh.model.mesh.generate(3)
    gmsh.write(str(OUTDIR / "mixedCells.cgns"))
    gmsh.finalize()


if __name__ == "__main__":
    OUTDIR.mkdir(parents=True, exist_ok=True)
    generate_single_tet()
    generate_cube_3d()
    generate_cavity_2d()
    generate_mixed_cells()
    print(f"Generated meshes in {OUTDIR}")
