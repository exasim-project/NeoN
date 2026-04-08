.. SPDX-FileCopyrightText: 2026 NeoN authors
..
.. SPDX-License-Identifier: MIT

Operator Assembly
=================

This page documents how the finite-volume operators ``gaussGreenGrad``,
``gaussGreenDiv``, and ``gaussGreenLaplacian`` assemble contributions into
field vectors and linear systems.  It is a companion to
:doc:`matrixAddressing` which explains the CSR/COO indexing used throughout.

Source files:

- ``src/finiteVolume/cellCentred/operators/gaussGreenGrad.cpp``
- ``src/finiteVolume/cellCentred/operators/gaussGreenDiv.cpp``
- ``src/finiteVolume/cellCentred/operators/gaussGreenLaplacian.cpp``

Conventions
-----------

Face orientation
    The face-area vector :math:`\mathbf{S}_f` points from the **owner** cell
    to the **neighbour** cell.  By construction, ``owner[f] < neighbour[f]``
    for all internal faces.

Interpolation weight
    ``weightsV[f]`` (alias ``w``) is the weight for the **owner** cell, so the
    face-centred value is

    .. math::

       \varphi_f = w \, \varphi_\text{own} + (1-w) \, \varphi_\text{nei}.

    Weights are produced by ``SurfaceInterpolation::weight()``; for linear
    interpolation this is the geometric inverse-distance weight biased toward
    the owner.

Matrix entry naming
    See :ref:`naming-inversion` in :doc:`matrixAddressing` for the known
    inversion between the helper-function names ``upperIdx``/``lowerIdx`` and
    standard LDU convention.  Inside the operator kernels the same inversion
    is reflected in the variable names ``valueUpper``/``valueLower`` and the
    offset arrays ``neiOffs``/``ownOffs``:

    - ``neiOffs[f]`` = ``neighbourOffset[f]`` → position of **A[nei,own]**
      (lower-triangular entry) within the neighbour's row.
    - ``ownOffs[f]`` = ``ownerOffset[f]`` → position of **A[own,nei]**
      (upper-triangular entry) within the owner's row.

    .. todo::

       Rename ``valueUpper``/``valueLower`` in operator kernels and
       ``upperIdx``/``lowerIdx`` in ``FaceToMatrixAddress`` to match standard
       LDU convention: *upper* = A[own,nei] (row < col), *lower* = A[nei,own]
       (row > col).  Cross-ref: ``include/NeoN/linearAlgebra/faceToMatrixAddress.hpp``.

Assembly calling order
    For implicit operators: internal faces → physical boundary faces.
    The physical boundary loop runs over ``facei ∈ [nInternalFaces, sGamma.size())``.

gaussGreenGrad — Explicit Only
------------------------------

``GaussGreenGrad`` provides only an explicit (field-to-field) gradient.
There is no implicit (matrix-assembling) overload; calling
``dsl::solve`` with a grad term will raise an error at runtime.

Algorithm
^^^^^^^^^

1. Interpolate ``phi`` to faces: ``phif = surfInterp.interpolate(phi)``.
2. Loop over internal faces ``i ∈ [0, nInternalFaces)``::

       Vec3 flux = faceAreaS[i] * phif[i];    // S_f * φ_f (vector)
       grad[owner[i]]    += flux;
       grad[neighbour[i]] -= flux;

3. Loop over boundary faces ``i ∈ [nInternalFaces, phif.size())``::

       Vec3 flux = faceAreaS[i] * phif[i];
       grad[own] += flux;                      // owner only, no neighbour

4. Normalise: ``grad[celli] *= operatorScaling[celli] / vol[celli]``.

The result is the standard Green–Gauss gradient:

.. math::

   (\nabla \varphi)_C \approx \frac{1}{V_C} \sum_f \mathbf{S}_f \, \varphi_f.

gaussGreenDiv — Explicit
------------------------

``computeDivExp`` interpolates ``phi`` to faces and calls the shared
``computeDiv`` helper.

Internal faces
^^^^^^^^^^^^^^

::

    ValueType flux = faceFlux[i] * phif[i];
    div[owner[i]]     += flux;
    div[neighbour[i]] -= flux;

Boundary faces
^^^^^^^^^^^^^^

::

    ValueType flux = faceFlux[i] * phif[i];
    div[own] += flux;           // owner contribution only

After both loops: ``div[celli] *= operatorScaling[celli] / vol[celli]``.

gaussGreenDiv — Implicit
------------------------

``computeDivImp`` assembles the implicit divergence operator into a
``LinearSystem``.  The face flux ``φ_f`` is decomposed via interpolation
weights into owner and neighbour contributions.

Internal faces
^^^^^^^^^^^^^^

Let ``φ = faceFlux[f]`` (signed scalar flux through face ``f``).  The
face-interpolated transported quantity is:

.. math::

   \varphi_f = w \, \varphi_\text{own} + (1-w) \, \varphi_\text{nei}

so the divergence contribution from face ``f`` to the owner row is
``+φ · φ_f = +φ(w φ_own + (1-w) φ_nei)``, giving matrix entries:

.. list-table:: Implicit divergence — internal face entries
   :header-rows: 1
   :widths: 25 30 20 25

   * - Matrix entry
     - Value
     - Code variable
     - Code line
   * - A[nei, own]  (lower tri.)
     - :math:`-w \cdot \varphi \cdot \text{scalingNei}`
     - ``valueUpper`` at ``neiOffs[f]``
     - ``values[rowNeiStart + neiOffs[facei]]``
   * - diag[own]
     - :math:`+w \cdot \varphi \cdot \text{scalingOwn}`
     - via ``atomic_sub(-valueUpper)``
     - ``atomic_sub(&values[diagOffs[own]], valueUpper)``
   * - A[own, nei]  (upper tri.)
     - :math:`+(1-w) \cdot \varphi \cdot \text{scalingOwn}`
     - ``valueLower`` at ``ownOffs[f]``
     - ``values[rowOwnStart + ownOffs[facei]]``
   * - diag[nei]
     - :math:`-(1-w) \cdot \varphi \cdot \text{scalingNei}`
     - via ``atomic_sub(valueLower)``
     - ``atomic_sub(&values[diagOffs[nei]], valueLower)``

.. note::

   **Non-conservative vs. conservative form.**
   NeoN assembles the *physical* (non-conservative) form.  OpenFOAM
   applies ``negSumDiag()`` after setting the off-diagonals, which
   forces every row to sum to zero (conservative form).  Concretely,
   for a single face the NeoN diagonals are:

   .. math::

      \text{diag[own]} &= +w\varphi, \\
      \text{diag[nei]} &= -(1-w)\varphi,

   whereas OpenFOAM produces:

   .. math::

      \text{diag[own]} &= -(1-w)\varphi, \\
      \text{diag[nei]} &= +w\varphi.

   The difference per face is :math:`+\varphi` on ``own`` and
   :math:`-\varphi` on ``nei``.  For incompressible flow the
   cell-wise divergence is zero (:math:`\sum_f \varphi_f = 0`), so the
   extra terms sum to zero in every row and both forms yield the same
   linear-system solution.

.. todo::

   The variable ``valueUpper`` (line ~206 of ``gaussGreenDiv.cpp``) is
   placed at ``neiOffs[facei]`` (= ``neighbourOffset``), which is
   position **A[nei,own]** — the *lower*-triangular entry.  Similarly
   ``valueLower`` is placed at ``ownOffs[facei]`` (= ``ownerOffset``),
   which is **A[own,nei]** — the *upper*-triangular entry.  The names
   are inverted relative to standard LDU convention and should be
   swapped together with the helper-function rename described above.

Physical boundary
^^^^^^^^^^^^^^^^^

The boundary loop runs over faces ``facei ∈ [nInternalFaces, faceFluxV.size())``.

Boundary weight: ``flux = bweights[bcfacei] * faceFlux[facei]``,
where ``bweights`` is the boundary interpolation weight for the owner cell.

Mixed-BC decomposition uses ``valueFraction[bcfacei]`` (:math:`f_D`):

- :math:`f_D = 1` → pure Dirichlet (``fixedValue``): boundary value
  :math:`\varphi_b = \text{refValue}` is fully prescribed, no
  implicit diagonal coupling.
- :math:`f_D = 0` → pure Neumann: gradient is prescribed
  (:math:`\nabla_n \varphi = \text{refGradient}`), full diagonal
  coupling.

.. list-table:: Implicit divergence — physical BC entries
   :header-rows: 1
   :widths: 25 40 35

   * - Contribution
     - Value
     - Code
   * - diag[own]
     - :math:`+\text{flux} \cdot \text{scalingOwn} \cdot (1-f_D)`
     - ``atomic_add(&values[diagOffs[own]], valueMat)``
   * - rhs[own]
     - :math:`-\text{flux} \cdot \text{scalingOwn} \cdot (f_D \cdot \text{refValue} + (1-f_D) \cdot \text{refGrad} / \delta)`
     - ``atomic_sub(&rhs[own], valueRhs)``

where :math:`\delta` is ``deltaCoeffs[bcfacei]`` = inverse distance from
cell centre to boundary face centre.

.. note::

   The diagonal term uses the **Neumann fraction** :math:`(1-f_D)`.  For
   pure Dirichlet the diagonal is unchanged and the full contribution is
   in the RHS; for pure Neumann the diagonal receives the flux and the
   RHS receives the gradient source.

gaussGreenLaplacian — Explicit
------------------------------

``computeLaplacianExp`` delegates face-normal gradients to a
``FaceNormalGradient`` helper and accumulates:

Internal faces
^^^^^^^^^^^^^^

::

    ValueType flux = faceArea[i] * fnGrad[i];   // |S_f| * (∇_n φ)_f
    laplacian[owner[i]]     += flux;
    laplacian[neighbour[i]] -= flux;

Boundary faces
^^^^^^^^^^^^^^

::

    ValueType flux = faceArea[i] * fnGrad[i];
    laplacian[own] += flux;

After both loops: ``laplacian[celli] *= operatorScaling[celli] / vol[celli]``.

gaussGreenLaplacian — Implicit
------------------------------

``computeLaplacianImpl`` assembles the diffusion operator
:math:`-\nabla \cdot (\gamma \nabla)` into the linear system.

Internal faces
^^^^^^^^^^^^^^

Face diffusion coefficient: ``flux = deltaCoeffs[facei] * sGamma[facei] * magFaceArea[facei]``,
i.e. :math:`\delta_f \cdot \gamma_f \cdot |S_f|`.

.. list-table:: Implicit Laplacian — internal face entries
   :header-rows: 1
   :widths: 25 30 20 25

   * - Matrix entry
     - Value
     - Note
     - Code
   * - A[nei, own]  (lower tri.)
     - :math:`+\text{flux} \cdot \text{scalingNei}`
     - at ``neiOffs[f]``; comment says "upper" (inverted)
     - ``values[rowNeiStart + neiOffs[facei]]``
   * - diag[own]
     - :math:`-\text{flux} \cdot \text{scalingOwn}`
     - ``atomic_sub``
     - ``atomic_sub(&values[diagOffs[own]], flux)``
   * - A[own, nei]  (upper tri.)
     - :math:`+\text{flux} \cdot \text{scalingOwn}`
     - at ``ownOffs[f]``; comment says "lower" (inverted)
     - ``values[rowOwnStart + ownOffs[facei]]``
   * - diag[nei]
     - :math:`-\text{flux} \cdot \text{scalingNei}`
     - ``atomic_sub``
     - ``atomic_sub(&values[diagOffs[nei]], flux)``

The assembled matrix is symmetric and negative semi-definite for
positive :math:`\gamma` and consistent operatorScaling, representing
:math:`-\nabla \cdot (\gamma \nabla \varphi)`.

.. note::

   **Matches OpenFOAM.**  OpenFOAM's ``gaussLaplacianScheme`` sets
   ``upper() = deltaCoeffs * gammaMagSf`` (positive) and calls
   ``negSumDiag()`` (diagonal = −sum of off-diagonals), yielding the
   same structure.

.. todo::

   The comments "``// add neighbour contribution upper``" (line ~118) and
   "``// add owner contribution lower``" (line ~132) in
   ``gaussGreenLaplacian.cpp`` use the same inverted upper/lower
   convention as the div kernel.  They should read "lower triangular
   A[nei,own]" and "upper triangular A[own,nei]" respectively.  Fix
   together with the ``upperIdx``/``lowerIdx`` rename.

Physical boundary
^^^^^^^^^^^^^^^^^

The boundary loop covers ``facei ∈ [nInternalFaces, sGamma.size())``.

Face diffusion coefficient (no ``deltaCoeffs`` here, unlike internal faces):

.. code-block:: text

    flux = sGamma[facei] * magFaceArea[facei]     // γ · |S_f|

Mixed-BC decomposition with ``valueFraction`` (:math:`f_D`):

.. list-table:: Implicit Laplacian — physical BC entries
   :header-rows: 1
   :widths: 25 45 30

   * - Contribution
     - Value
     - Code
   * - diag[own]
     - :math:`-\text{flux} \cdot \text{scalingOwn} \cdot f_D \cdot \delta_f`
     - ``atomic_sub(&values[diagOffs[own]], valueMat)``
   * - rhs[own]
     - :math:`-\text{flux} \cdot \text{scalingOwn} \cdot (f_D \cdot \delta_f \cdot \text{refValue} + (1-f_D) \cdot \text{refGradient})`
     - ``atomic_sub(&rhs[own], valueRhs)``

where :math:`\delta_f` = ``deltaCoeffs[facei]`` (all-faces array index,
consistent throughout the boundary kernel).

The diagonal term uses the **Dirichlet fraction** :math:`f_D`:

- Pure Dirichlet (:math:`f_D = 1`): ``diag[own] -= flux * δ``,
  providing full implicit coupling to :math:`\varphi_C`.
- Pure Neumann (:math:`f_D = 0`): no diagonal change; gradient source
  enters via the RHS only.

.. note::

   The boundary ``flux = γ · |S_f|`` lacks the ``deltaCoeffs`` factor
   present in the internal face flux (``δ · γ · |S_f|``).  The
   ``deltaCoeffs`` factor is applied explicitly in ``valueMat`` and
   ``valueRhs`` instead, so the total implicit coupling is still
   ``δ · γ · |S_f| · f_D``.  This differs from the internal face
   assembly where ``flux`` already absorbs ``δ``.
