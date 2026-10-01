Under-Relaxation
----------------

Under-relaxation stabilises iterative solvers by dampening updates between
outer iterations.  NeoN provides two complementary forms: **matrix
(equation) relaxation** and **field (explicit) relaxation**.

Matrix Under-Relaxation
~~~~~~~~~~~~~~~~~~~~~~~

Matrix relaxation modifies the assembled linear system in-place before
solving so that the solution is pulled only partially toward the new
iterate.  Given an under-relaxation factor ``alpha ∈ (0, 1)``, it
strengthens the diagonal for numerical stability and adds a source
correction so the fixed point of the relaxed system equals the fixed point
of the unrelaxed one.

**API:** ``dsl::applyMatrixRelaxation(ls, solution, alpha)``

The function operates on the augmented diagonal — boundary contributions
are permanently baked into ``matrix.values[diagIdx(cell)]`` at assembly
time.  Per cell the kernel computes:

.. code-block:: text

   D_aug     = matrix.values[diagIdx(cell)]           (boundary already included)
   D_dom     = max(|D_aug|, sumMagOffDiag[cell])       (dominance clamp)
   D_relaxed = copySign(D_dom / alpha, D_aug)          (scale by 1/alpha, keep sign)
   matrix.values[diagIdx(cell)] = D_relaxed
   rhs[cell] += (D_relaxed - D_aug) * psi_prev[cell]  (source correction)

The dominance clamp ensures diagonal dominance is not reduced below the
off-diagonal magnitude sum.  ``componentCopySign`` preserves the
negative-diagonal sign convention so downstream operations such as
``rAU = V / diag`` remain correct for ``alpha < 1``.

The source correction ``(D_relaxed - D_aug) * psi_prev`` cancels exactly at
the converged solution, so the fixed point is unchanged.

``alpha <= 0`` or ``alpha == 1`` is a bitwise no-op.

The implementation uses a single fused kernel: the off-diagonal sum,
dominance clamp, diagonal update, and source correction are computed in
one pass over each cell's CSR row.  There is no scratch allocation.

Field (Explicit) Under-Relaxation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Field relaxation blends the current field value toward the previous outer
iteration, independently of the linear system.  It is typically applied
after the momentum equation is solved and before extracting ``rAU``/``HbyA``
(SIMPLE/PISO outer loop).

**API:** ``dsl::applyFieldRelaxation(solution, previous, alpha)``

.. code-block:: text

   psi[c] = prev[c] + alpha * (psi[c] - prev[c])

Only the internal vector is blended; boundary conditions must be
re-evaluated by the caller via ``solution.correctBoundaryConditions()``.

``alpha <= 0`` or ``alpha == 1`` is a bitwise no-op.

**Snapshot helper:** ``dsl::fieldRelaxationSnapshot(field)``

Returns an executor-correct deep copy of ``field.internalVector()`` to be
stored as the start-of-iteration snapshot and passed to
``applyFieldRelaxation``.  Subsequent writes to ``field`` do not affect the
snapshot.

Typical usage pattern:

.. code-block:: cpp

   // At the start of each outer iteration, snapshot the current field.
   auto prevU = dsl::fieldRelaxationSnapshot(U);

   // ... solve momentum equation ...

   // Apply field relaxation.
   dsl::applyFieldRelaxation(U, prevU, alphaU);
   U.correctBoundaryConditions();
