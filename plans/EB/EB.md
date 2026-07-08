# Embedded Boundary (EB) Support for NeoN/blockamr

## Summary

Integrate AMReX's Embedded Boundary (EB2) infrastructure into the NeoN/blockamr
Python framework, enabling cut-cell geometry (cylinders, spheres, arbitrary
implicit functions) in incompressible Navier-Stokes solves.  The final
deliverable is a **flow-past-cylinder example** (`example/blockamr/flow_past_cylinder.py`)
analogous to `double_shear_layer.py`, with quantitative verification against
known drag coefficients at Re = 20 and Re = 100.

### What exists today

| Layer | Status |
|-------|--------|
| AMReX EB2 C++ (geometry, `EBFabFactory`, volume/area fractions, `MLEBABecLap`, `MLEBTensorOp`) | Available in `amrex/Src/EB/` and `amrex/Src/LinearSolvers/MLMG/` |
| IAMReX reference solver (NS + EB + AMR) | `IAMReX/Source/` — working tutorials in `Tutorials/FlowPastCylinder/` |
| NeoN C++ bindings (`linop.cpp`) | `MLPoisson`, `MLABecLaplacian`, `MLNodeLaplacian`, `MLMG` bound — **no EB operators** |
| NeoN Python solver | `DSLIncompressibleSolver` — periodic/Dirichlet/Neumann, no EB awareness |

### What needs to be built

1. **C++ nanobind bindings** for EB2 geometry, `EBFArrayBoxFactory`, EB linear
   operators, and EB utility functions. Guarded with `#ifdef AMREX_USE_EB` so
   non-EB AMReX builds still compile.
2. **Runtime-switchable EB on existing types** — extend `Mesh` / `AmrMesh` /
   `CellField` with an *optional* `EBFArrayBoxFactory` per level. No parallel
   `EBMesh` / `EBCellField` classes; `mesh.has_eb` drives all branching.
3. **EB-aware paths inside the existing solver** — `DSLIncompressibleSolver`
   gains a `_make_abec_linop(mesh, lev)` helper that returns `MLEBABecLaplacian`
   or `MLABecLaplacian` depending on `mesh.has_eb`. Covered-cell zeroing,
   wall BCs on the EB surface, and `volfrac` masking are conditional one-liners.
4. **Verification example** — flow past cylinder with drag/lift computation.

### Design principle: one solver, two modes

AMReX's EB API permits **runtime** EB/non-EB switching when compiled with
`AMReX_EB=ON`:

| Layer | Polymorphism | Consequence |
|---|---|---|
| `MultiFab` factory | `FabFactory<FArrayBox>` base — `DefaultFabFactory` or `EBFArrayBoxFactory` chosen at construction | A single `MultiFab` type holds both. No EB MultiFab class. |
| Linear operators | `MLABecLaplacian` vs `MLEBABecLaplacian` are siblings of `MLLinOp` | Type-level branch at construction; downstream `MLMG` / `setBCoeffs` / `solve` / `getFluxes` are identical. |
| `Hydro::MacProjector` | Inspects the factory and constructs the right linop internally | One call site, both modes — exactly how IAMReX uses it (`IAMReX/Source/MacProj.cpp:1115`). |
| EB utilities (`EB_set_covered`, `EB_interp_*`) | Free functions | Cheap to guard with `if mesh.has_eb`. |

The **only unavoidable type-level branch** is choosing the linop class. It is
hidden inside one helper. Everything else (fields, MLMG setup, BC application,
flux extraction) is shared.

When `AMReX_EB=OFF`, `mesh.has_eb` is always `False`, the EB binding symbols
do not exist, and `_make_abec_linop` only ever returns the non-EB class. The
Python solver code is unchanged.

---

## Architecture

```
Python DSL layer
─────────────────────────────────────────────────
  DSLIncompressibleSolver  (single class, EB optional)
    ├── CellField (U, p)        ── fill_patch zeros covered cells iff mesh.has_eb
    ├── FaceField (phi)         ── zero flux on EB faces iff mesh.has_eb
    ├── Mesh / AmrMesh          ── optional EBFArrayBoxFactory per level
    ├── _make_abec_linop()      ── returns MLEBABecLaplacian or MLABecLaplacian
    ├── MAC projection          ── MLMG on whichever linop the helper returned
    ├── Pressure solve          ── same helper (cell-centred recommended)
    ├── Viscous diffusion       ── _make_tensor_linop() → MLEBTensorOp or MLTensorOp
    └── EB redistribution       ── only invoked when mesh.has_eb
─────────────────────────────────────────────────
nanobind C++ bindings  (src/bindings/blockAMR/, all #ifdef AMREX_USE_EB)
─────────────────────────────────────────────────
  eb2.cpp        EB2 geometry: IndexSpace, Build, implicit functions
  ebfactory.cpp  EBFArrayBoxFactory, volfrac, areafrac, facecent
  linop.cpp      +MLEBABecLaplacian, +MLEBTensorOp
  eb_utils.cpp   EB_set_covered, EB redistribution
─────────────────────────────────────────────────
AMReX C++
```

### Solver step (single path; EB calls are conditional)

```python
def step(self):
    eb = self.mesh.has_eb

    # 1. Fill ghost cells; CellField.fill_patch zeros covered cells when eb
    for lev in range(n_levels):
        U.fill_patch(lev, t)

    # 2. Cell → face interpolation
    interpolate(U, phi)
    if eb:
        for lev in range(n_levels):
            eb_zero_covered_faces(phi, lev)      # zero flux through EB

    # 3. MAC projection — _mac_project picks the right linop via helper
    self._mac_project(phi)                       # MLEBABecLap or MLABecLap

    # 4. Momentum predictor; volfrac multiplied unconditionally
    #    (volfrac is all-ones when no EB)
    solve(ddt(U) + div(phi, U) - laplacian(nu, U), t, dt)
    if eb:
        for lev in range(n_levels):
            eb_set_covered(U.mf[lev], 0.0)
            eb_redistribute(U.mf[lev], lev)

    # 5. Pressure correction
    self._pressure_solve(U)                      # same helper, same call site
    U -= dt * grad(p)

    # 6. Post-step
    if eb:
        for lev in range(n_levels):
            eb_set_covered(U.mf[lev], 0.0)
    t += dt
```

---

## Milestones

### M0 — CMake: enable EB in AMReX build

**Goal**: Ensure AMReX is compiled with `AMReX_EB=ON` by default. The runtime
EB/non-EB switch only works when this is on; when off, NeoN still builds and
`mesh.has_eb` is always `False`.

- [ ] Add `AMReX_EB=ON` to the CPM fetch for AMReX in `cmake/CxxThirdParty.cmake`.
- [ ] Expose `NeoN_WITH_AMREX_EB` (default ON) so size-constrained builds can
      opt out.
- [ ] Verify headers `AMReX_EB2.H`, `AMReX_EBFabFactory.H`,
      `AMReX_MLEBABecLap.H` are available after build.
- [ ] Propagate `AMREX_USE_EB` to `src/bindings/blockAMR/` so `#ifdef`-guarded
      TUs see it.

**Verification**: `cmake --build --preset develop` succeeds; `import neon.blockamr;
neon.blockamr.has_eb_support()` returns `True`. Building with
`-DNeoN_WITH_AMREX_EB=OFF` also succeeds and the same call returns `False`.

---

### M1 — C++ bindings: EB2 geometry

**Files**: `src/bindings/blockAMR/eb2.cpp` (new), update `module.cpp` and
`bindings.hpp`.

Bind the following from `AMReX_EB2.H` and `AMReX_EB2_IF.H`:

| C++ class / function | Python name | Notes |
|----------------------|-------------|-------|
| `EB2::SphereIF` | `EB2_SphereIF(center, radius, has_fluid_inside)` | |
| `EB2::CylinderIF` | `EB2_CylinderIF(radius, direction, center, has_fluid_inside)` | |
| `EB2::PlaneIF` | `EB2_PlaneIF(point, normal)` | |
| `EB2::BoxIF` | `EB2_BoxIF(lo, hi, has_fluid_inside)` | |
| `EB2::makeUnion` | `eb2_union(a, b)` | Template — may need type-erased wrapper |
| `EB2::makeIntersection` | `eb2_intersection(a, b)` | |
| `EB2::makeComplement` | `eb2_complement(a)` | |
| `EB2::Build(gshop, geom, max_level, max_coarsening)` | `eb2_build(gshop, geom, ...)` | Creates `IndexSpace` |
| `EB2::makeShop<IF>` | `eb2_make_shop(implicit_func)` | |

**Design note**: AMReX implicit functions are templated.  The simplest
approach is to create a type-erased `ImplicitFuncWrapper` on the C++ side
that stores an `std::function<Real(AMREX_D_DECL(Real,Real,Real))>` and
satisfies the `EB2` concept.  Alternatively, bind each concrete IF and the
corresponding `makeShop` overload.

**Verification**:
```python
import neon.blockamr as blockamr
with blockamr.runtime():
    cyl = blockamr.EB2_CylinderIF(0.05, 2, [0.15, 0.2, 0.0], False)
    gshop = blockamr.eb2_make_shop(cyl)
    blockamr.eb2_build(gshop, geom, 0, 100)
    # No crash; IndexSpace is populated
```

---

### M2 — C++ bindings: EBFArrayBoxFactory & EB data

**Files**: `src/bindings/blockAMR/ebfactory.cpp` (new).

| C++ | Python | Returns |
|-----|--------|---------|
| `makeEBFabFactory(geom, ba, dm, ng, EBSupport::full)` | `make_eb_factory(geom, ba, dm)` | `EBFArrayBoxFactory` |
| `factory.getVolFrac()` | `factory.vol_frac()` | `MultiFab` (1 comp, cell-centred) |
| `factory.getAreaFrac()` | `factory.area_frac(dir)` | `MultiFab` (1 comp, face-centred) |
| `factory.getFaceCent()` | `factory.face_cent(dir)` | `MultiFab` |
| `factory.getMultiEBCellFlagFab()` | `factory.cell_flags()` | Expose `FabType` queries |
| `EB_set_covered(mf, val)` | `eb_set_covered(mf, val)` | In-place |
| `EB_set_covered_faces(mf_arr, val)` | `eb_set_covered_faces(phi_face, val)` | |

**Verification**:
```python
ebf = blockamr.make_eb_factory(geom, ba, dm)
vf = ebf.vol_frac()
# vf is a MultiFab with values in [0, 1]; sum > 0 and < total cells
assert 0 < vf.sum() < N_cells**3
```

---

### M3 — C++ bindings: EB linear operators

**Files**: extend `src/bindings/blockAMR/linop.cpp`.

| Operator | Key methods to bind |
|----------|-------------------|
| `MLEBABecLaplacian(geoms, bas, dms, info, {ebfactories})` | Constructor, `setScalars`, `setACoeffs`, `setBCoeffs`, `setEBDirichlet`, `setEBHomogDirichlet` |
| `MLEBTensorOp(geoms, bas, dms, info, {ebfactories})` | Constructor, `setShearViscosity`, `setBulkViscosity`, `setEBDirichlet`, `setEBHomogDirichlet` |

Both work with the existing `MLMG` solver binding.

**Verification**:
```python
lp = blockamr.MLEBABecLaplacian([geom], [ba], [dm], info, [ebf])
lp.set_scalars(0.0, 1.0)
lp.set_eb_homog_dirichlet(0)
mlmg = blockamr.MLMG(lp)
mlmg.solve([phi], [rhs], 1e-10, 1e-12)
# Converges in finite iterations
```

---

### M4 — Python: EB as an *option* on existing Mesh / CellField

**No new mesh or field classes.** Extend `Mesh` / `AmrMesh` / `CellField` in
place so a single solver call site handles both modes.

**Files**: update `mesh.py`, `field.py`. New helper module `src/neon/blockamr/eb.py`
holds only the EB-specific *free functions* (geometry helpers, volfrac
allocation, ones-fallback).

```python
class Mesh:
    def __init__(self, ba, dm, geom, eb_factory=None):
        ...
        # _factories[lev] is either EBFArrayBoxFactory or DefaultFabFactory.
        # Both satisfy FabFactory<FArrayBox>; MultiFab construction is uniform.
        self._factories = [eb_factory or blockamr.default_fab_factory()]

    @property
    def has_eb(self) -> bool:
        return any(blockamr.is_eb_factory(f) for f in self._factories)

    def factory(self, lev=0):
        return self._factories[lev]

    def vol_frac(self, lev=0):
        """Always returns a MultiFab. All-ones when no EB on this level."""
        ...

class AmrMesh(Mesh):
    # _on_new_level builds an EBFArrayBoxFactory per level when an
    # EB2::IndexSpace exists for that Geometry, otherwise DefaultFabFactory.
    ...

class CellField:
    def fill_patch(self, lev, time):
        ...  # existing fill
        if self.mesh.has_eb:
            blockamr.eb_set_covered(self.mf[lev], 0.0)
```

Construction in user code is uniform — no `EBMesh`:

```python
ebf = blockamr.make_eb_factory(geom, ba, dm)   # or None
mesh = blockamr.Mesh(ba, dm, geom, eb_factory=ebf)
```

**Verification**:
- `Mesh(..., eb_factory=None)`: `mesh.has_eb is False`, `vol_frac()` is
  all ones, `CellField.fill_patch` skips `eb_set_covered`.
- `Mesh(..., eb_factory=ebf_cyl)`: `mesh.has_eb is True`, `vol_frac().sum()`
  strictly less than total cells, covered cells become 0 after `fill_patch`.

---

### M5 — Python: MAC projection helper that picks the linop

**Files**: update `dsl_solver.py` only — **no `eb_solver.py`**.

Add a single private helper that returns the right linop. Both classes share
the same downstream API (`setScalars`, `setBCoeffs`, `MLMG.solve`,
`getFluxes`), so the rest of `_mac_project_level` is unchanged.

```python
def _make_abec_linop(self, lev):
    geom, ba, dm = self.mesh.geom(lev), self.mesh.ba(lev), self.mesh.dm(lev)
    info = blockamr.LPInfo()
    if self.mesh.has_eb:
        ebf = self.mesh.factory(lev)   # EBFArrayBoxFactory
        return blockamr.MLEBABecLaplacian([geom], [ba], [dm], info, [ebf])
    return blockamr.MLABecLaplacian([geom], [ba], [dm], info)

def _mac_project_level(self, phi, lev):
    lp = self._make_abec_linop(lev)
    lp.set_scalars(0.0, 1.0)
    lp.set_b_coeffs(0, bx, by, bz)
    if self.mesh.has_eb:
        lp.set_eb_homog_dirichlet(0)   # no-slip on EB surface
    mlmg = blockamr.MLMG(lp)
    mlmg.solve(...)
    mlmg.get_fluxes(...)
    if self.mesh.has_eb:
        for d in range(3):
            blockamr.eb_set_covered_faces(phi[lev][d].mf, 0.0)
```

The non-EB path is byte-identical to today's `_mac_project_level`, so this
change is a strict refactor that adds an EB branch — no regression risk for
existing periodic / Dirichlet examples.

**Verification**:
- Existing `double_shear_layer.py` produces bit-identical results before/after
  this refactor (no EB constructed).
- New cylinder case: divergence of corrected phi is O(solver tolerance)
  everywhere in fluid cells.

---

### M6 — Python: pressure solve (same helper)

Use the **same** `_make_abec_linop` helper from M5. The cell-centred
formulation is recommended precisely because `MLABecLaplacian` and
`MLEBABecLaplacian` share an identical setup API — the solver code is
literally one routine for both modes:

```python
def _pressure_solve(self, U_star):
    for lev in range(self.mesh.n_levels()):
        lp = self._make_abec_linop(lev)
        lp.set_scalars(0.0, 1.0)
        lp.set_b_coeffs(0, dt_fx, dt_fy, dt_fz)
        if self.mesh.has_eb:
            lp.set_eb_homog_dirichlet(0)
        mlmg = blockamr.MLMG(lp)
        mlmg.solve([self.p.mf[lev]], [rhs[lev]], rtol, atol)
```

**Design decision**: IAMReX uses a nodal projection. A cell-centred
projection with `MLABec` / `MLEBABec` is simpler and — critically — gives a
single code path for both modes. Recommended for the first implementation.

**Verification**: After pressure correction, `max(abs(div(U)))` in fluid
cells < atol, in both EB and non-EB runs.

---

### M7 — Python: explicit operators always multiply by volfrac

**Key idea**: `volfrac` is provided unconditionally. When the mesh has no EB,
`mesh.vol_frac(lev)` returns an all-ones MultiFab created once at mesh
construction. The JAX/Pallas stencil kernel always multiplies by it — there
are **no two code paths** in the hot kernel, and no runtime branch.

**Option A — Mask-based (simplest)**:
```python
# In Div.build_kernel_3d (single path, EB or not):
result = div_result * volfrac[box_id, i, j, k]
```
- No EB: `volfrac == 1` everywhere → mathematically equivalent to today.
- EB: covered cells get 0; cut cells get a reduced (first-order) contribution.

**Option B — Cut-cell flux redistribution (second-order)**:
Requires `areafrac` and a redistribution step. Implement only when M8 shows
M7-A is insufficient for the target Reynolds numbers.

**Recommended path**: Start with Option A. Confirm bit-equivalence on
`double_shear_layer.py` (volfrac all-ones) before moving on.

**Verification**:
- `double_shear_layer.py` numerics unchanged after M7 (regression check).
- Cylinder case: uniform field → `div == 0` in fluid; `laplacian` of linear
  field == 0 in fluid.

---

### M8 — Example: flow past cylinder

**File**: `example/blockamr/flow_past_cylinder.py`

```python
"""Flow past a cylinder — incompressible Navier-Stokes with EB.

Solves the incompressible NS equations on a [0, 1.2] x [0, 0.4] x [0, 0.1]
domain with a cylinder of radius 0.05 at (0.15, 0.2):

    ddt(U) + div(phi, U) - laplacian(nu, U) = 0
    div(U) = 0

BCs:  inflow (U=1,0,0) at x=0, outflow at x=1.2,
      no-slip walls at y=0 and y=0.4, periodic in z.
      No-slip on the cylinder surface (EB).

Supports single-level and AMR (--max-level).

Usage:
    python example/blockamr/flow_past_cylinder.py --re 20
    python example/blockamr/flow_past_cylinder.py --re 100 --max-level 1
"""

import argparse
import neon.blockamr as blockamr
from neon.blockamr.dsl_solver import DSLIncompressibleSolver


def run(Re=20, N=96, max_level=0, ...):
    nu = 1.0 / Re
    D = 0.1  # cylinder diameter

    # --- geometry ---
    box = blockamr.Box([0, 0, 0], [3*N-1, N-1, N//12-1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.2, 0.4, 0.1])
    geom = blockamr.Geometry(box, rb, 0, [0, 0, 1])  # periodic only in z

    # --- EB: cylinder ---
    cyl = blockamr.EB2_CylinderIF(0.05, 2, [0.15, 0.2, 0.0], False)
    gshop = blockamr.eb2_make_shop(cyl)
    blockamr.eb2_build(gshop, geom, max_level, 100)

    # --- mesh: same Mesh class, EB enabled via factory ---
    ba = blockamr.BoxArray(box)
    ba.max_size(32)
    dm = blockamr.DistributionMapping(ba)
    ebf = blockamr.make_eb_factory(geom, ba, dm)
    mesh = blockamr.Mesh(ba, dm, geom, eb_factory=ebf)
    assert mesh.has_eb

    # --- BCs ---
    from neon.blockamr.bc import DirichletBC, NeumannBC, VectorBC, FillPatchWithBC
    U_bc = VectorBC(
        lo=[DirichletBC([1, 0, 0]), DirichletBC([0, 0, 0]), None],   # inflow, wall, periodic
        hi=[NeumannBC(),            DirichletBC([0, 0, 0]), None],   # outflow, wall, periodic
    )

    # --- solver: same DSLIncompressibleSolver as double_shear_layer.py ---
    solver = DSLIncompressibleSolver(
        mesh, nu, dt=cfl * (1.2/N),
        fill_patch=FillPatchWithBC(U_bc),
        schemes_p={"rtol": 0, "atol": 1e-8, "max_iter": 200},
    )

    # --- initial condition: uniform flow ---
    for lev in range(mesh.n_levels()):
        solver.U.mf[lev].set_val(0.0)
        # set x-component to 1.0
        # ...

    # --- time loop ---
    for step in range(1, n_steps + 1):
        solver.step()
        if step % plot_interval == 0:
            cd, cl = compute_drag_lift(solver, D, nu)
            print(f"Step {step}: Cd={cd:.4f}, Cl={cl:.4f}")
            solver.write_plotfile(f"plt_cyl_{step:05d}")

    # --- verification ---
    cd, cl = compute_drag_lift(solver, D, nu)
    verify_drag(Re, cd)
```

---

## Verification Strategy

### Unit tests (per milestone)

| Test | What it checks |
|------|---------------|
| `test_mesh_no_eb_default` | `Mesh(...)` without `eb_factory`: `has_eb is False`, `vol_frac` all-ones |
| `test_eb2_build` | EB2 geometry creation doesn't crash; `IndexSpace` is populated |
| `test_eb_factory_volfrac` | Volume fractions: sum < total cells, all in [0,1], covered cells = 0 |
| `test_mesh_with_eb` | `Mesh(..., eb_factory=ebf)`: `has_eb is True`, `vol_frac().sum()` < total |
| `test_eb_set_covered` | After `eb_set_covered(mf, 0)`, all covered cells are exactly 0 |
| `test_cellfield_fillpatch_eb` | `CellField.fill_patch` zeros covered cells iff `mesh.has_eb` |
| `test_make_abec_linop_dispatch` | Helper returns `MLABecLaplacian` for non-EB mesh, `MLEBABecLaplacian` for EB mesh |
| `test_eb_mac_projection` | After MAC project, `max(abs(div(phi)))` in fluid cells < 1e-10 |
| `test_eb_pressure_solve` | After pressure correction, `max(abs(div(U)))` < atol |
| `test_eb_laplacian_linear` | Laplacian of `f(x) = x` through EB domain: result = 0 in fluid cells |
| `test_eb_conservation` | Total momentum in fluid cells is conserved (within solver tolerance) |
| `test_dsl_no_regression` | `double_shear_layer.py` produces bit-identical output before/after the M5–M7 refactor |

### Integration / regression tests

| Test | Expected |
|------|----------|
| **Re = 20, steady** | C_d = 2.00 +/- 0.05 (Tritton 1959), C_l ~ 0 (symmetric) |
| **Re = 100, unsteady** | C_d ~ 1.33 +/- 0.05, Strouhal St ~ 0.164 +/- 0.01 (Williamson 1996) |
| **Grid convergence** | C_d converges with refinement; Richardson extrapolation gives order >= 1 |
| **AMR vs uniform** | AMR (max_level=1) result within 2% of uniform at equivalent resolution |

### Drag / lift computation

```python
def compute_drag_lift(solver, D, nu):
    """Compute drag and lift coefficients from pressure and viscous stress on EB.

    C_d = F_x / (0.5 * rho * U_inf^2 * D)
    C_l = F_y / (0.5 * rho * U_inf^2 * D)

    Forces can be computed by:
    (a) Integrating pressure and viscous stress on EB surface
        (requires EB surface normals and area fractions)
    (b) Control volume approach: momentum flux through a box around the cylinder
    """
    # Option (b) is easier to implement initially:
    # F = -d/dt(integral rho*U dV) + integral (rho*U*U + p*I - tau) . n dS
    # On a steady-state flow, d/dt term = 0
    ...
```

### Strouhal number (Re = 100)

```python
def compute_strouhal(cl_history, dt, D, U_inf=1.0):
    """FFT of C_l time series to find dominant frequency."""
    from jax.numpy.fft import rfft, rfftfreq
    spectrum = jnp.abs(rfft(jnp.array(cl_history)))
    freqs = rfftfreq(len(cl_history), d=dt)
    f_peak = freqs[jnp.argmax(spectrum[1:]) + 1]
    return f_peak * D / U_inf
```

---

## Dependencies & risks

| Risk | Mitigation |
|------|-----------|
| AMReX not compiled with `AMReX_EB=ON` | M0 addresses this first; fail-fast cmake check |
| Templated implicit functions hard to bind | Type-erased wrapper or bind concrete types only |
| Nodal pressure solve with EB is complex | Use cell-centred `MLEBABecLap` instead |
| Small cut cells cause CFL restriction | Redistribution (M7 Option B) or merge small cells |
| JAX kernels need per-cell volfrac access | Pass volfrac as additional kernel input buffer |
| Performance regression from EB masking | Profile; EB overhead is mainly in linear solvers (C++) |

---

## File inventory (new / modified)

| File | Action |
|------|--------|
| `cmake/CxxThirdParty.cmake` | Modify — add `AMReX_EB=ON`, `NeoN_WITH_AMREX_EB` option |
| `src/bindings/blockAMR/eb2.cpp` | **New** — EB2 geometry bindings (`#ifdef AMREX_USE_EB`) |
| `src/bindings/blockAMR/ebfactory.cpp` | **New** — EBFArrayBoxFactory bindings (`#ifdef AMREX_USE_EB`) |
| `src/bindings/blockAMR/eb_utils.cpp` | **New** — `EB_set_covered`, redistribution (`#ifdef AMREX_USE_EB`) |
| `src/bindings/blockAMR/linop.cpp` | Modify — add `MLEBABecLaplacian`, `MLEBTensorOp` under `#ifdef` |
| `src/bindings/blockAMR/module.cpp` | Modify — register EB bindings conditionally; expose `has_eb_support()` |
| `src/bindings/blockAMR/bindings.hpp` | Modify — declare new registration functions |
| `src/neon/blockamr/mesh.py` | Modify — add optional `eb_factory`, `has_eb`, `factory(lev)`, `vol_frac(lev)` |
| `src/neon/blockamr/field.py` | Modify — `CellField.fill_patch` calls `eb_set_covered` when `mesh.has_eb` |
| `src/neon/blockamr/dsl_solver.py` | Modify — add `_make_abec_linop` helper; existing `_mac_project_level`, `_pressure_solve` gain EB branches |
| `src/neon/blockamr/eb.py` | **New** — *free functions only* (geometry helpers, ones-fallback volfrac); no new mesh/field classes |
| `src/neon/blockamr/__init__.py` | Modify — re-export EB free functions and `has_eb_support` |
| `example/blockamr/flow_past_cylinder.py` | **New** — example |
| `test/blockamr/test_eb.py` | **New** — unit tests for EB option on `Mesh`/`CellField` |
| `test/blockamr/test_dsl_solver_no_regression.py` | **New** — bit-equivalence of `double_shear_layer.py` after the M5/M7 refactor |
| `test/blockamr/test_flow_past_cylinder.py` | **New** — regression test |

### Files explicitly NOT created

`EBMesh`, `EBAmrMesh`, `EBCellField`, `EBFaceField`, and
`DSLEBIncompressibleSolver` are **not** created. Keeping a single class
hierarchy is what makes runtime EB switching work and prevents drift between
parallel solver paths.
