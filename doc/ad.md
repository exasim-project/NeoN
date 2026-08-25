# Automatic differentiation in NeoN

Status: **experimental**, forward mode only.

## Scope of this branch

This branch establishes the forward-mode vertical slice: declare design
variables, propagate derivatives through ordinary arithmetic, read exact
sensitivities. It is deliberately the smallest complete path from user
declaration to verified gradient.

**In scope:** scalar and small-array design variables (model coefficients,
operating conditions, parameterisation coefficients).

**Out of scope, deliberately:**

- *Mesh coordinates.* Geometric quantities (`mesh.V()`, `mesh.Sf()`, face
  distances, non-orthogonality corrections) remain plain `NeoN::scalar`. Any
  attempt to differentiate through mesh metrics is therefore a compile error
  rather than a silently wrong sensitivity. This also keeps the Jacobian
  sparsity pattern fixed and leaves the skewness corrections passive.
- *Reverse mode.* Needed once `n_alpha` reaches `O(n_face)` or `O(n_cell)`. The
  registry contract is written to be mode-agnostic so `gradient()` keeps its
  signature when sensitivities start coming from an adjoint sweep.

## Why forward mode, and why not a tape

Forward mode costs one primal evaluation per design variable and requires no
history. Reverse mode costs one per functional but must traverse the primal
backwards.

Tape-based AD (CoDiPack, dco, Adept) records an execution trace at scalar-
operation granularity. On GPU this fails on five independent counts: unbounded
dynamic allocation in device code; tape size of order 10^9 entries for a single
LES timestep; a global append cursor that serialises thread writes; a reverse
sweep of random-access atomics; and an active type whose layout breaks
coalescing and downstream `double*` interfaces.

`Dual<T, N>` avoids all of these by fixing `N` at compile time: the type is
trivially copyable, has no pointers, and allocates nothing.

## Layout note

`Dual` is Array-of-Struct. For `N <= ~4` this is generally the better device
choice since a cell's derivatives are consumed with its value. For larger `N`,
Struct-of-Array with a hidden derivative dimension (cf. Sacado's Kokkos View
support) should be **measured before committing** — coalescing behaviour of the
derivative dimension is where most GPU AD attempts quietly lose 5-10x.

## Blockers found in the current tree

1. `NeoN::scalar` is a hard typedef in `core/primitives/scalar.hpp`. Operators
   and fields must be templated on `ValueType` before `Dual` can flow through
   the FVM stack.
2. `dsl::Coeff` stores `scalar` and returns `scalar` from `operator[]`. This is
   the immediate blocker for making a diffusion coefficient a design variable.
3. `core/primitives/traits.hpp` uses *function* templates. C++ forbids partial
   specialization of these, so each `Dual<T, N>` instantiation must be
   registered by hand via `NeoN_DUAL_REGISTER_TRAITS`. The correct fix is to
   convert the traits to structs, which partially specialize cleanly. That
   touches every existing primitive and is left out of this MWE.
4. MPI reductions hardcode `MPI_DOUBLE`; a `Dual` reduction needs a custom
   datatype and op.

## Differentiability audit — not yet done

Comparisons on `Dual` act on the primal only, so a branch taken on a comparison
is not differentiated. The following need an explicit decision (frozen vs.
smoothed) before results are trusted:

- upwind and linearUpwind switching
- the limited-corrected `snGrad` limiter
- `min`/`max` in limiters
- `sqrt` near zero (`ROOTVSMALL` guards)
- wall functions

## Verification

`examples/ad/standalone/forwardSensitivity.cpp` solves 1D steady diffusion and
compares AD gradients against central finite differences for three design
variables and two functionals.

```
cd examples/ad/standalone
g++ -std=c++20 -Ishim -I../../../include forwardSensitivity.cpp -o forwardSensitivity
./forwardSensitivity
```

The `shim/` directory contains a minimal stand-in for `Kokkos_Core.hpp` so the
example builds with no Kokkos installation. It is never on the include path of a
normal NeoN build.

Unit tests: `test/core/primitives/dual.cpp`, including the shared-leaf diamond
case — one design variable reaching the functional by two paths. Getting the
accumulation wrong there yields a partial gradient that looks entirely
plausible, which is why it is tested explicitly.

Finite differences are only a first-line check; they share the primal code path
and cannot detect an error common to both. Once reverse mode exists, the
tangent-adjoint dot-product identity

    <J_bar, J_dot> = <alpha_bar, alpha_dot>

holds to machine precision and should become the primary test, applied operator
by operator. Note it holds only where the primal is genuinely differentiable —
the switches listed above will break it at switching points.

## Next steps

1. Convert `traits.hpp` to struct traits; remove the registration macro.
2. Template `dsl::Coeff` on `ValueType`.
3. Template one operator end-to-end (`gaussGreenLaplacian` is the natural
   choice) and run the MWE through the real DSL rather than a Thomas solve.
4. Replace direct differentiation of the solve with the implicit function
   theorem against Ginkgo — differentiating through solver iterations is both
   wrong and expensive.
