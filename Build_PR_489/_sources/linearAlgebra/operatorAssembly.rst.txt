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

    Therefore :math:`F_f > 0` means flux **leaves** the owner cell and
    **enters** the neighbour cell.  This sign convention is only valid because
    :math:`\mathbf{S}_f` points from owner to neighbour.

Interpolation weight
    ``weightsV[f]`` (alias ``w``) is the weight for the **owner** cell, so the
    face-centred value is

    .. math::

       \varphi_f = w \, \varphi_\text{own} + (1-w) \, \varphi_\text{nei}.

    Weights are produced by ``SurfaceInterpolation::weight()``; for linear
    interpolation this is the geometric inverse-distance weight biased toward
    the owner.

Matrix entry naming
    Following the LDU convention (see :doc:`matrixAddressing`):

    - ``upperIdx(own, f)`` = ``rowOffs[own] + ownerOffset[f]`` → position of
      :math:`A[\text{own},\text{nei}]` (**upper**-triangular, own < nei).
    - ``lowerIdx(nei, f)`` = ``rowOffs[nei] + neighbourOffset[f]`` → position of
      :math:`A[\text{nei},\text{own}]` (**lower**-triangular, nei > own).

    Operator kernels use the direct offset form for performance:

    .. code-block:: cpp

        // A[own, nei] — upper triangular
        values[rowOwnStart + ownOffs[facei]] ...
        // values[matIt->upperIdx(own, facei)] ...   // equivalent via matrixIterator

        // A[nei, own] — lower triangular
        values[rowNeiStart + neiOffs[facei]] ...
        // values[matIt->lowerIdx(nei, facei)] ...   // equivalent via matrixIterator

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

       Vec3 flux = faceAreaS[i] * phif[i];    // S_f * φ_f (S_f points owner→nei by construction)
       grad[owner[i]]    += flux;              // +S_f * φ_f (S_f is outward from owner)
       grad[neighbour[i]] -= flux;             // −S_f * φ_f (S_f is inward to neighbour)

3. Loop over boundary faces ``i ∈ [nInternalFaces, phif.size())``::

       Vec3 flux = faceAreaS[i] * phif[i];
       grad[own] += flux;                      // +S_f * φ_f (S_f outward from owner; no neighbour)

4. Normalise: ``grad[celli] *= operatorScaling[celli] / vol[celli]``.

The result is the standard Green–Gauss gradient:

.. math::

   (\nabla \varphi)_C \approx \frac{1}{V_C} \sum_f \mathbf{S}_f \, \varphi_f.

``S_f`` points from owner to neighbour, so it is the outward area vector for the owner and
the inward area vector for the neighbour.  The subtract for the neighbour follows directly.

gaussGreenDiv — Explicit
------------------------

``computeDivExp`` interpolates ``phi`` to faces and calls the shared
``computeDiv`` helper.  The result is the **positive** mathematical divergence
:math:`+\nabla \cdot (F\varphi)`.

.. note::

   **Consistent sign convention.**
   Both the explicit ``computeDiv`` and the implicit ``computeDivImp`` assemble
   :math:`+\nabla \cdot (F\varphi)` — the positive divergence operator — matching
   OpenFOAM's ``fvm::div`` convention.  The DSL equation form mirrors OpenFOAM:

   .. code-block:: cpp

      dsl::solve(ddt(phi) + div(faceFlux, phi) - laplacian(gamma, phi) == source, ...)

Internal faces
^^^^^^^^^^^^^^

::

    ValueType flux = faceFlux[i] * phif[i];    // F_f * φ_f
    div[owner[i]]     += flux;                  // F_f is outward from owner (S_f points owner→nei)
    div[neighbour[i]] -= flux;                  // F_f is inward to neighbour → subtract

Sign rationale:

- ``F_f > 0`` means flux leaves the owner cell (S_f points from owner to neighbour by construction).
- The **divergence** at owner is the net outward flux: a positive outward :math:`F_f` gives a
  *positive* divergence contribution at the owner cell. This is why ``div[own] += flux`` is correct
  — it does not contradict the fact that the owner is losing :math:`\varphi`.  The conservation
  equation uses :math:`\partial\varphi/\partial t = -\nabla\cdot(F\varphi)`, where the sign flip
  is applied externally (or in the implicit operator).
- For the neighbour, :math:`F_f` is inward (S_f points away from neighbour), so the outward flux
  from the neighbour's perspective is :math:`-F_f`, hence ``div[nei] -= flux``.

Boundary faces
^^^^^^^^^^^^^^

::

    ValueType flux = faceFlux[i] * phif[i];
    div[own] += flux;           // F_f outward from owner (no neighbour on this rank)

After both loops: ``div[celli] *= operatorScaling[celli] / vol[celli]``.

gaussGreenDiv — Implicit
------------------------

``computeDivImp`` assembles :math:`+\nabla \cdot (F\varphi)` into a ``LinearSystem``,
identical in sign to ``computeDivExp`` and to OpenFOAM's ``fvm::div``.

Divergence convention
^^^^^^^^^^^^^^^^^^^^^

With :math:`F_f > 0` meaning flux from owner P to neighbour N
(valid because :math:`\mathbf{S}_f` points from owner to neighbour):

- **Owner row** P: total contribution **+F_f**
  (:math:`\mathbf{S}_f` is outward from P → positive divergence at P).
- **Neighbour row** N: total contribution **−F_f**
  (:math:`\mathbf{S}_f` is inward to N from this face → negative contribution at N).

Global verification: :math:`+F_f - F_f = 0` per face ✓

Internal faces
^^^^^^^^^^^^^^

The face-interpolated transported quantity is:

.. math::

   \varphi_f = w \, \varphi_\text{own} + (1-w) \, \varphi_\text{nei}

The face flux :math:`F_f \varphi_f` decomposes linearly into owner and neighbour contributions:

.. math::

   F_f \, \varphi_f
     = w \, F_f \, \varphi_\text{own}
     + (1-w) F_f \, \varphi_\text{nei}

In the code the variables ``ownFluxContrib`` and ``neiFluxContrib`` carry a
**pre-applied minus sign** so that uniform ``+=``/``-=`` patterns in the assembly
loops still produce the correct positive divergence:

.. code-block:: cpp

   ownFluxContrib = -w * F_f           // note: NEGATIVE; code uses this in += and atomic_sub
   neiFluxContrib = -(1-w) * F_f       // note: NEGATIVE; code uses this in -= and atomic_add

.. list-table:: Implicit divergence — internal face entries
   :header-rows: 1
   :widths: 22 30 22 26

   * - Matrix entry
     - Value
     - Sign note
     - Code
   * - A[nei, own]  (lower tri.)
     - :math:`-w \cdot F_f \cdot \text{scalingNei}`
     - own value → negative contribution to nei's divergence (inward face for N)
     - ``values[rowNeiStart + neiOffs[f]] += ownFluxContrib * scalingNei``
   * - diag[own]
     - :math:`+w \cdot F_f \cdot \text{scalingOwn}`
     - own value → positive contribution to P's divergence (outward face for P)
     - ``atomic_sub(&values[diagOwn], ownFluxContrib * scalingOwn)``
   * - A[own, nei]  (upper tri.)
     - :math:`+(1-w) \cdot F_f \cdot \text{scalingOwn}`
     - nei value → positive contribution to P's divergence
     - ``values[rowOwnStart + ownOffs[f]] -= neiFluxContrib * scalingOwn``
   * - diag[nei]
     - :math:`-(1-w) \cdot F_f \cdot \text{scalingNei}`
     - nei value → negative contribution to N's divergence (inward face for N)
     - ``atomic_add(&values[diagNei], neiFluxContrib * scalingNei)``

Row-sum verification per face:

- Owner row: :math:`+w F_f + (1-w) F_f = +F_f` ✓
- Neighbour row: :math:`-w F_f - (1-w) F_f = -F_f` ✓

.. note::

   The diagonal is **assembled explicitly** — there is no separate ``negSumDiag``
   pass.  Each face contributes :math:`+w F_f` to the owner diagonal and
   :math:`-(1-w) F_f` to the neighbour diagonal, so the full sums converge to
   :math:`+F_f` and :math:`-F_f` respectively.

.. note::

   **Matches OpenFOAM.**  OpenFOAM's ``gaussConvectionScheme`` sets
   ``lower[f] = -w F_f``, ``upper[f] = (1-w) F_f``, then ``negSumDiag()`` gives
   ``diag[own] += w F_f`` and ``diag[nei] -= (1-w) F_f`` — the **same structure**
   as NeoN.

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
``FaceNormalGradient`` helper and accumulates the **positive** Laplacian
:math:`+\nabla \cdot (\gamma \nabla \varphi)`.

Internal faces
^^^^^^^^^^^^^^

``fnGrad[f]`` is computed by ``FaceNormalGradient`` as:

.. code-block:: text

    fnGrad[f] = nonOrthDeltaCoeffs[f] * (phi[nei] − phi[own])

``fnGrad > 0`` when :math:`\varphi_N > \varphi_P` (gradient points outward from owner, i.e.
S_f direction).  In this case, diffusion brings :math:`\varphi` *into* the owner cell and
*out of* the neighbour cell, which is the correct physical interpretation (diffusion flows
from high to low concentration).

::

    ValueType flux = faceArea[i] * fnGrad[i];   // |S_f| * (∂φ/∂n)_f   (outward from owner)
    laplacian[owner[i]]     += flux;             // fnGrad>0: φ_N>φ_P, owner gains φ → positive
    laplacian[neighbour[i]] -= flux;             // neighbour loses φ → negative

Boundary faces
^^^^^^^^^^^^^^

::

    ValueType flux = faceArea[i] * fnGrad[i];
    laplacian[own] += flux;                      // S_f outward from owner; no neighbour on this rank

After both loops: ``laplacian[celli] *= operatorScaling[celli] / vol[celli]``.

gaussGreenLaplacian — Implicit
------------------------------

``computeLaplacianImpl`` assembles the diffusion operator
:math:`+\nabla \cdot (\gamma \nabla \varphi)` into the linear system.

Internal faces
^^^^^^^^^^^^^^

Face diffusion coefficient: ``flux = deltaCoeffs[facei] * sGamma[facei] * magFaceArea[facei]``,
i.e. :math:`\delta_f \cdot \gamma_f \cdot |S_f|`.

The Laplacian is symmetric — the same ``flux`` value enters both the owner and neighbour rows
with appropriate signs.  :math:`\mathbf{S}_f` points from owner to neighbour, so
diffusion out of one cell equals diffusion into the other.

.. list-table:: Implicit Laplacian — internal face entries
   :header-rows: 1
   :widths: 25 30 22 23

   * - Matrix entry
     - Value
     - Sign note
     - Code
   * - A[nei, own]  (lower tri.)
     - :math:`+\text{flux} \cdot \text{scalingNei}`
     - enters neighbour → positive
     - ``values[rowNeiStart + neiOffs[facei]]``
   * - diag[own]
     - :math:`-\text{flux} \cdot \text{scalingOwn}`
     - leaves owner → negative
     - ``atomic_sub(&values[diagOffs[own]], flux)``
   * - A[own, nei]  (upper tri.)
     - :math:`+\text{flux} \cdot \text{scalingOwn}`
     - enters owner → positive
     - ``values[rowOwnStart + ownOffs[facei]]``
   * - diag[nei]
     - :math:`-\text{flux} \cdot \text{scalingNei}`
     - leaves neighbour → negative
     - ``atomic_sub(&values[diagOffs[nei]], flux)``

The assembled matrix is symmetric and negative semi-definite for
positive :math:`\gamma` and consistent ``operatorScaling``, representing
:math:`-\nabla \cdot (\gamma \nabla \varphi)`.

.. note::

   **Matches OpenFOAM.**  OpenFOAM's ``gaussLaplacianScheme`` sets
   ``upper() = deltaCoeffs * gammaMagSf`` (positive) and calls
   ``negSumDiag()`` (diagonal = −sum of off-diagonals), yielding the
   same structure.

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
