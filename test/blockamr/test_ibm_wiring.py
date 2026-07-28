# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""The IBM wiring — resolver, band flow and the non-fluid pin (B5, B6, B7).

The layering checkpoint of ``plans/IBM/sessions.md`` W3: an evaluate that
resolves a boundary scheme, builds its rows and applies them after the interior
sweep, with **nothing method-specific written**. Everything below therefore
runs on a method and a boundary scheme registered *by the test*, which is the
strongest available evidence that the production path knows nothing about any
particular method.

**Levels, and why.** The resolver and the band flow are exercised through
``evaluate`` (``plans/IBM/verification.md`` §1: the equation surface is the
suite's vocabulary) — a missing pair, an undeclared stencil shape and a row
landing on a cell are all observable there. The non-fluid pin's *arithmetic* is
not: it writes a field in place, and routing it through a laplacian would test
the laplacian — so those tests call ``mesh.ibm.pin_non_fluid`` directly, the way
``test_ibm_band_table.py`` and ``test_ibm_mesh.py`` already do for the layers
below the equation. Its **placement** is observable at the equation surface, and
that is what the B25 tests below assert: the pin is a classification write, so
the first ``evaluate`` of a field pins it and no later one writes it at all
(design §7, ``plans/IBM/review.md`` §4 Q3).

**The numbers are exact.** The field is ``T = x^2`` on a mesh with
``dx = 1/8``, so every value is a dyadic rational computed by hand in the
comment above the assert and compared with ``==``.
"""

import numpy as np
import pytest

import blockamr
from blockamr.dsl import Equation, evaluate, exp, solve
from blockamr.field import CellField, FaceField
from blockamr.ibm import BandRows, Cylinder, FixedValue
from blockamr.mesh import Mesh
from blockamr.operators.div import update_face_fluxes
from blockamr.schemes.boundary import BOUNDARY_SCHEMES

# The backend every rung runs on (verification plan §3 spells "cpp" throughout).
BACKEND = "cpp"

N = 8  # cells per side; dx = 1/8, so x^2 at a cell centre is exact in binary64

# A cylinder whose surface passes between two rings of cell centres: the four
# columns at i, j in {5, 6} are non-fluid (their centres are 0.0884 from the
# axis, inside R = 0.1), and every other cell is fluid.
CENTRE = (0.75, 0.75)
R = 0.1
SOLID_COLUMNS = {(5, 5), (5, 6), (6, 5), (6, 6)}

# The name the test method registers under, and the cells its boundary scheme
# writes. Both are far from the body, so the row's donor is a fluid cell the
# pin never touches.
METHOD = "testRows"
TARGET = (2, 2, 2)
DONOR = (3, 2, 2)

#: phi(3, 2, 2) = ((3 + 0.5)/8)^2 = 0.4375^2
DONOR_VALUE = 0.19140625

#: ``(term class name, band width)`` of every ``rows`` call, cleared per test.
ASKED_WIDTHS = []


# ---------------------------------------------------------------------------
# the test-registered method and its boundary schemes
# ---------------------------------------------------------------------------


class _RowsMethod:
    """An operator method on the band flow, with no code in ``src/``.

    It carries no preprocessing of its own: this session's flow reads the
    method-agnostic classification only, and a method that needs more declares
    its own data type (design §2.4) — which B8 does for ``ghostCell``.
    """

    name = METHOD
    kind = "operator"
    requires_bodies = True
    data_type = type(None)

    @staticmethod
    def preprocess(mesh, lev):
        return None


class _OneRowScheme:
    """A boundary scheme whose rows are written by hand, not derived.

    One row per level: ``out(TARGET) = coeff * (2 * phi(DONOR) + 7)``. The
    term's own coefficient scales it so two terms of the same operator produce
    different numbers, which is what makes their accumulation visible.

    Every ``rows`` call records the band width the driver asked it for, in
    :data:`ASKED_WIDTHS` — the equation's width, not the term's own (design §6,
    the composition rule). The driver builds a fresh scheme instance per
    evaluate, so the record is module-level and the fixture clears it.
    """

    operator = "laplacian"
    method = METHOD
    stride = 1

    def __init__(self, interior_scheme):
        self.interior = interior_scheme

    def rows(self, term, ibm, lev, ncomp, t, width):
        ASKED_WIDTHS.append((type(term).__name__, width))
        coeff = float(term.coeff)
        return BandRows(
            target=np.array([TARGET], dtype=np.int32),
            stencil=np.array([[DONOR]], dtype=np.int32),
            a=np.array([[2.0 * coeff]], dtype=np.float64),
            nnz=np.array([1], dtype=np.int32),
            c=np.full((1, ncomp), 7.0 * coeff, dtype=np.float64),
            patch=np.zeros(1, dtype=np.int32),
            box_offset=np.array([0, 1], dtype=np.int32),
            stride=1,
        )


class _NeverCalledScheme(_OneRowScheme):
    """The proof that an empty band does not launch the sweep."""

    def rows(self, term, ibm, lev, ncomp, t, width):
        raise AssertionError("the band sweep ran on an empty band")


class _WideDivScheme(_OneRowScheme):
    """The div peer of :class:`_OneRowScheme`, for the two-band case."""

    operator = "div"


class _NoShapeLaplacian:
    """An interior scheme that declares a width but no stencil shape."""

    type = "NoShapeLaplacian"
    stencil_width = 1


@pytest.fixture
def registered(monkeypatch):
    """Register the test method, and boundary schemes, for one test only.

    ``_METHODS`` is the IBM registry's own table (B13 owns making registration
    public); ``BOUNDARY_SCHEMES`` is public API (design.md §2.4). ``monkeypatch``
    restores both, so no test can see another's registrations.
    """
    import blockamr.ibm as ibm_registry

    ASKED_WIDTHS.clear()
    monkeypatch.setitem(ibm_registry._METHODS, METHOD, _RowsMethod)

    def _register(*scheme_classes):
        for scheme_cls in scheme_classes:
            monkeypatch.setitem(
                BOUNDARY_SCHEMES, (scheme_cls.operator, scheme_cls.method), scheme_cls
            )

    return _register


# ---------------------------------------------------------------------------
# helpers — mesh, field, results
# ---------------------------------------------------------------------------


def _mesh(bodies=None, n=N):
    """One box on the unit cube, periodic, ``n^3`` cells."""
    box = blockamr.Box([0, 0, 0], [n - 1, n - 1, n - 1])
    rb = blockamr.RealBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    geom = blockamr.Geometry(box, rb, 0, [1, 1, 1])
    ba = blockamr.BoxArray(box)
    ba.max_size(n)
    dm = blockamr.DistributionMapping(ba)
    mesh = Mesh(ba, dm, geom)
    mesh.bodies = {} if bodies is None else bodies
    return mesh


def _cylinder(centre=CENTRE):
    return {"cyl": Cylinder(centre=centre, radius=R, axis=2)}


def _quadratic_field(mesh, ncomp=1):
    """``T = x^2`` — exact in binary64 at every cell centre of this mesh."""
    T = CellField(mesh, ncomp=ncomp, ngrow=1, name="T", ibm_bc={"cyl": FixedValue(0.0)})
    mf = T.mf[0]
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_to_host(mfi)
        lo = mfi.valid_box().small_end()
        i = np.arange(arr.shape[0])[:, None, None] + lo[0]
        x = (i + 0.5) / N
        arr[:, :, :, :] = (x * x)[..., None]
        mf.copy_from(mfi, arr)
    T.fill_patch(0, 0.0)
    return T


def _uniform_flux(mesh, ngrow):
    ff = FaceField(mesh, ncomp=1, ngrow=ngrow, name="phi")
    update_face_fluxes(
        ff[0],
        lambda x, y, z, t: (np.ones_like(x), np.ones_like(x), np.ones_like(x)),
        mesh.geom(0),
        t=0.0,
    )
    return ff


def _valid_cells(field):
    """Bitwise snapshot of a field's valid cells (one box)."""
    mf = field.mf[0]
    return np.array(mf.copy_to_host(next(iter(blockamr.MFIterator(mf)))), copy=True)


def _result(out):
    """The single box's level-0 result as an ``(N, N, N)`` array."""
    arr = np.asarray(out[0][0])
    return arr.reshape(arr.shape[:3])


def _sol(method=None):
    """The fvSolution block: no ``"ibm"`` key at all means no IBM."""
    return {"backend": BACKEND} if method is None else {"ibm": method, "backend": BACKEND}


def _solid_columns():
    """Cells inside the cylinder, from the analytic body — an independent
    oracle, never the implementation's own classification (verification §10)."""
    mask = np.zeros((N, N, N), dtype=bool)
    for i, j in SOLID_COLUMNS:
        mask[i, j, :] = True
    return mask


def _seed_solid(field, value):
    """Write ``value`` into every solid column of ``field``'s valid cells.

    The probe for "did the pin run again": a value no pin and no sweep can
    produce, planted in the cells the pin owns.
    """
    mf = field.mf[0]
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_to_host(mfi)
        lo = mfi.valid_box().small_end()
        for i, j in SOLID_COLUMNS:
            arr[i - lo[0], j - lo[1], :, :] = value
        mf.copy_from(mfi, arr)


def _reads_a_pinned_cell():
    """Cells whose width-1 cross stencil touches a solid column.

    The pin changes those cells' operator value, so they are excluded from the
    "everything else is bitwise the plain operator" comparison — they are the
    band, which is exactly the region the rows own.
    """
    solid = _solid_columns()
    mask = solid.copy()
    for d in range(3):
        for step in (1, -1):
            mask |= np.roll(solid, step, axis=d)
    return mask


# ---------------------------------------------------------------------------
# the resolver (B5) — a missing capability is a sentence, never a wrong number
# ---------------------------------------------------------------------------


def test_missing_boundary_scheme_names_the_pair(blockamr_session, registered):
    """Verification §8. The equation asks for a laplacian under a method that
    has a div boundary scheme and nothing else; falling back to the interior
    scheme would drop the wall condition and return a plausible field, so the
    resolver refuses and says which pair is missing and which exist."""
    registered(_WideDivScheme)
    mesh = _mesh(bodies=_cylinder())
    T = _quadratic_field(mesh)

    with pytest.raises(ValueError) as excinfo:
        evaluate(Equation(exp.laplacian(1.0, T)), t=0.0, solution=_sol(METHOD))

    message = str(excinfo.value)
    assert "laplacian" in message
    assert METHOD in message
    assert "('div', 'testRows')" in message


def test_undeclared_stencil_shape_is_rejected(blockamr_session, registered):
    """Verification §8. ``band(w)`` is the cross-stencil band; a scheme that
    reads corners needs the Chebyshev depth instead, and taking the cross band
    for it under-selects along the diagonals — a wrong answer in the band with
    a correct bulk (design §4). A scheme that declares neither shape is
    therefore refused, naming itself and both shapes."""
    registered(_OneRowScheme)
    mesh = _mesh(bodies=_cylinder())
    T = _quadratic_field(mesh)
    eqn = Equation(exp.laplacian(1.0, T), schemes={"Laplacian": _NoShapeLaplacian()})

    with pytest.raises(ValueError) as excinfo:
        evaluate(eqn, t=0.0, solution=_sol(METHOD))

    message = str(excinfo.value)
    assert "NoShapeLaplacian" in message
    assert "cross" in message
    assert "box" in message


def test_every_registry_scheme_declares_a_stencil_shape(blockamr_session):
    """The declaration is what the resolver requires, so every scheme in
    ``SCHEME_REGISTRY`` must carry one — generated from the registry so a new
    scheme cannot be added without answering the question."""
    from blockamr.schemes.registry import SCHEME_REGISTRY

    for operator, table in SCHEME_REGISTRY.items():
        if operator == "ddt":  # a time scheme reads no neighbour cell
            continue
        for name, scheme_cls in table.items():
            assert scheme_cls().stencil_shape in ("cross", "box"), f"{operator}/{name}"


# ---------------------------------------------------------------------------
# the band flow (B6) — resolve -> rows -> table -> kernel
# ---------------------------------------------------------------------------


def test_the_bands_rows_overwrite_the_interior_result_and_nothing_else(
    blockamr_session, registered
):
    """The whole wiring, end to end, on hand-built rows.

    The boundary scheme writes ``out(2,2,2) = 2*phi(3,2,2) + 7``. With
    ``T = x^2`` and ``dx = 1/8`` that is ``2 * 0.4375^2 + 7 = 7.3828125``
    exactly — a number the laplacian could not produce, so the assertion sees
    the row and not the sweep.

    Everywhere else the result must be **bitwise** the plain operator's: the
    interior sweep is the same call with the same arguments as the no-IBM path,
    and the only cells excluded are the band's own (the pin changes what the
    sweep reads there, which is why the rows own them).

    The plain evaluate runs **first**, on a pristine field: the pin writes the
    solid cells of ``T``, and this test is not the place to assert whether that
    is observable (design §7 says it is idempotent, and W4 owns reconciling it
    with the purity test).
    """
    registered(_OneRowScheme)
    mesh = _mesh(bodies=_cylinder())
    T = _quadratic_field(mesh)
    eqn = Equation(exp.laplacian(1.0, T))

    plain = _result(evaluate(eqn, t=0.0, solution=_sol()))
    banded = _result(evaluate(eqn, t=0.0, solution=_sol(METHOD)))

    # 2 * 0.19140625 + 7
    assert banded[TARGET] == 7.3828125
    assert plain[TARGET] != 7.3828125, "the row value must not be what the sweep writes"

    untouched = ~_reads_a_pinned_cell()
    untouched[TARGET] = False
    np.testing.assert_array_equal(banded[untouched], plain[untouched])


def test_a_second_term_adds_its_rows_to_the_first_terms_band_value(blockamr_session, registered):
    """Two terms, one band: the first term's rows replace what the sweep left
    there and the second term's are **added** — the same accumulation the
    interior sweeps do into the scratch source.

    ``2 * exp.laplacian`` scales the row through the term's own coefficient, so
    the two contributions are distinguishable: ``7.3828125 + 14.765625``.
    Overwriting twice would leave 14.765625, adding twice would leave
    22.1484375 plus whatever the sweep wrote.
    """
    registered(_OneRowScheme)
    mesh = _mesh(bodies=_cylinder())
    T = _quadratic_field(mesh)
    eqn = Equation(exp.laplacian(1.0, T) + 2.0 * exp.laplacian(1.0, T))

    banded = _result(evaluate(eqn, t=0.0, solution=_sol(METHOD)))

    assert banded[TARGET] == 7.3828125 + 14.765625


def test_terms_of_different_widths_are_all_asked_for_the_equations_widest_band(
    blockamr_session, registered
):
    """A width-1 laplacian beside a width-2 div: **one** band, the widest.

    The band is a property of the equation, not of one term (design §6, the
    composition rule). Every term's rows are built over ``band(2)`` here, which
    is what makes 'the first term writes, the rest add' exact: a cell in
    ``band(2)`` carries the sum of both terms' rows, and a cell only the div's
    band contains still carries the laplacian's own interior value — supplied
    by its row rather than left behind by the sweep.

    Asserted on the width the driver asks for, because that *is* the contract
    between the driver and a boundary scheme; the numbers it produces are
    ``test_ibm_combinations.py``'s hand-computed mixed-width case.
    """
    registered(_OneRowScheme, _WideDivScheme)
    mesh = _mesh(bodies=_cylinder())
    T = _quadratic_field(mesh)
    eqn = Equation(
        exp.laplacian(1.0, T) + exp.div(_uniform_flux(mesh, T.ngrow), T),
        schemes={"Div": "quick"},
    )

    evaluate(eqn, t=0.0, solution=_sol(METHOD))

    assert ASKED_WIDTHS == [("Laplacian", 2), ("Div", 2)]


def test_an_empty_band_is_bitwise_identical_to_no_ibm(blockamr_session, registered):
    """Rung 2, through the band flow. A body outside the domain has no boundary
    cell on this mesh, so there is nothing to correct and the sweep must not
    launch at all: the boundary scheme raises if it is ever asked for rows, and
    the result is compared with ``assert_array_equal`` — a tolerance here would
    permit exactly the coupling this forbids (verification §10)."""
    registered(_NeverCalledScheme)
    mesh = _mesh(bodies=_cylinder(centre=(99.0, 99.0)))
    T = _quadratic_field(mesh)
    eqn = Equation(exp.laplacian(1.0, T))

    plain = _result(evaluate(eqn, t=0.0, solution=_sol()))
    far = _result(evaluate(eqn, t=0.0, solution=_sol(METHOD)))

    np.testing.assert_array_equal(far, plain)


def test_an_empty_band_leaves_the_field_unpinned(blockamr_session, registered):
    """The other half of the short-circuit: with nothing to correct, the pin
    does not run either, so an evaluate under the method is bitwise
    indistinguishable from one without it — on the *field* as well as on the
    result."""
    registered(_NeverCalledScheme)
    mesh = _mesh(bodies=_cylinder(centre=(99.0, 99.0)))
    T = _quadratic_field(mesh)
    before = _valid_cells(T)

    evaluate(Equation(exp.laplacian(1.0, T)), t=0.0, solution=_sol(METHOD))

    np.testing.assert_array_equal(_valid_cells(T), before)


# ---------------------------------------------------------------------------
# a pointwise term in the band (B41) — the composition rule's degenerate case
# ---------------------------------------------------------------------------

# These two run on the *real* ``ghostCell`` method rather than the hand-built
# rows above: the claim is about what ``source x ghostCell`` emits, and the row
# is derived from the two general rules rather than invented (decision Q23/P1,
# ``plans/IBM/review.md`` §4). A pointwise term has no stencil and so no band of
# its own; by the composition rule its row everywhere is its plain interior
# formula, and by the non-fluid convention a solid cell is ``nnz = 0, c = 0``.


def _source_field(mesh):
    """``S = y`` — exact in binary64 at every cell centre, and nonzero in every
    solid column, so 'the source wrote nothing' and 'the source wrote 0' are
    not the same observation."""
    S = CellField(mesh, ncomp=1, ngrow=1, name="S")
    mf = S.mf[0]
    for mfi in blockamr.MFIterator(mf):
        arr = mf.copy_to_host(mfi)
        lo = mfi.valid_box().small_end()
        j = np.arange(arr.shape[1])[None, :, None] + lo[1]
        arr[:, :, :, :] = ((j + 0.5) / N)[..., None]
        mf.copy_from(mfi, arr)
    S.fill_patch(0, 0.0)
    return S


def test_a_source_terms_band_row_is_its_plain_interior_value(blockamr_session):
    """In the band the source contributes exactly ``coeff * S`` — the same value
    its interior sweep writes everywhere else.

    The band cells' result is *overwritten* by the first term's rows and added
    to by the rest, so a source term that emitted no rows would have its sweep
    contribution silently erased on precisely the cells the accuracy study
    measures. Comparing a ``laplacian`` evaluate with a ``laplacian + source``
    one isolates that contribution, and the two are compared bitwise: the row's
    ``c`` is ``S`` to the last bit and the apply adds it once.
    """
    mesh = _mesh(bodies=_cylinder())
    T = _quadratic_field(mesh)
    S = _source_field(mesh)

    plain = _result(evaluate(Equation(exp.laplacian(1.0, T)), t=0.0, solution=_sol("ghostCell")))
    with_source = _result(
        evaluate(
            Equation(exp.laplacian(1.0, T) + exp.source(S)),
            t=0.0,
            solution=_sol("ghostCell"),
        )
    )

    band = _reads_a_pinned_cell() & ~_solid_columns()
    assert band.any(), "the probe would be blind"
    s = _valid_cells(S)[..., 0]
    np.testing.assert_array_equal(with_source[band], plain[band] + s[band])


def test_a_source_term_writes_nothing_into_a_non_fluid_cell(blockamr_session):
    """The non-fluid convention (design §7), unchanged by a pointwise term: a
    ``depth <= 0`` cell is a row with ``nnz = 0, c = 0``.

    ``S`` is nonzero in every solid column, so a source that ignored the
    classification would leave its own value there — read by nothing and
    plotted by everything."""
    mesh = _mesh(bodies=_cylinder())
    T = _quadratic_field(mesh)
    S = _source_field(mesh)
    solid = _solid_columns()
    assert (_valid_cells(S)[..., 0][solid] != 0.0).all(), "the probe would be blind"

    with_source = _result(
        evaluate(
            Equation(exp.laplacian(1.0, T) + exp.source(S)),
            t=0.0,
            solution=_sol("ghostCell"),
        )
    )

    np.testing.assert_array_equal(with_source[solid], 0.0)


# ---------------------------------------------------------------------------
# the non-fluid pin (B7)
# ---------------------------------------------------------------------------


def test_pinning_writes_the_pin_value_into_every_non_fluid_cell(blockamr_session):
    """Design §7. The interior sweep reads non-fluid neighbours at a band cell,
    so those cells must hold a finite value rather than whatever was there.
    The pin is ``IbmGeometry.non_fluid_pin``, 0.0 by default, and ``T = x^2``
    is nonzero in every one of the cells concerned — so 'was pinned' and 'was
    already 0' are not the same observation."""
    mesh = _mesh(bodies=_cylinder())
    T = _quadratic_field(mesh)
    solid = _solid_columns()
    assert (_valid_cells(T)[..., 0][solid] != 0.0).all(), "the probe would be blind"

    mesh.ibm.pin_non_fluid(T, 0)

    np.testing.assert_array_equal(_valid_cells(T)[..., 0][solid], 0.0)


def test_pinning_leaves_every_fluid_cell_bitwise_untouched(blockamr_session):
    """The pin is the only write this architecture makes to a user field, and
    its licence is that no fluid cell is in it: every row is ``nnz = 0`` over a
    ``depth <= 0`` target."""
    mesh = _mesh(bodies=_cylinder())
    T = _quadratic_field(mesh)
    before = _valid_cells(T)

    mesh.ibm.pin_non_fluid(T, 0)

    fluid = ~_solid_columns()
    np.testing.assert_array_equal(_valid_cells(T)[..., 0][fluid], before[..., 0][fluid])


def test_pinning_twice_is_bitwise_a_no_op(blockamr_session):
    """Idempotence is what keeps ``evaluate`` pure in the sense the suite
    asserts: after the first pin the field is bitwise unchanged by every
    further one (design §7)."""
    mesh = _mesh(bodies=_cylinder())
    T = _quadratic_field(mesh)

    mesh.ibm.pin_non_fluid(T, 0)
    once = _valid_cells(T)
    mesh.ibm.pin_non_fluid(T, 0)

    np.testing.assert_array_equal(_valid_cells(T), once)


def test_pinning_a_body_free_mesh_writes_nothing(blockamr_session):
    """No body, no non-fluid cell, no table and no launch."""
    mesh = _mesh()
    T = _quadratic_field(mesh)
    before = _valid_cells(T)

    mesh.ibm.pin_non_fluid(T, 0)

    np.testing.assert_array_equal(_valid_cells(T), before)


def test_pinning_a_vector_field_pins_every_component(blockamr_session):
    """The pin table carries one constant per component, so a vector field is
    the same rows with a wider ``c`` — not a second table format."""
    mesh = _mesh(bodies=_cylinder())
    T = _quadratic_field(mesh, ncomp=3)

    mesh.ibm.pin_non_fluid(T, 0)

    np.testing.assert_array_equal(_valid_cells(T)[_solid_columns()], 0.0)


# ---------------------------------------------------------------------------
# the pin at classification time (B25) — once per (field, method, lev,
# grid_version), not once per evaluate (design §7, review.md §4 Q3)
# ---------------------------------------------------------------------------


def test_evaluate_pins_the_solid_cells_of_a_field_it_has_not_seen(blockamr_session, registered):
    """The first evaluate of a field *is* its classification event in v1 (Q3):
    nothing below the driver ever takes a field, so the pin is applied when the
    driver is built for it. ``T = x^2`` is nonzero in every solid cell before
    the evaluate, so 'was pinned' and 'was already 0' are not the same
    observation."""
    registered(_OneRowScheme)
    mesh = _mesh(bodies=_cylinder())
    T = _quadratic_field(mesh)
    solid = _solid_columns()
    assert (_valid_cells(T)[..., 0][solid] != 0.0).all(), "the probe would be blind"

    evaluate(Equation(exp.laplacian(1.0, T)), t=0.0, solution=_sol(METHOD))

    np.testing.assert_array_equal(_valid_cells(T)[..., 0][solid], 0.0)


def test_the_pin_does_not_run_again_in_a_later_evaluate(blockamr_session, registered):
    """The once-per-key observable (design §7, review.md §4 Q3). The pin belongs
    to the classification, not to the evaluate: after the field's first
    evaluate, no further evaluate writes it — not even a solid cell someone
    dirtied in between. Seeding 7.0 into the solid columns and finding it
    **bitwise** there after a second evaluate is the whole claim; a per-evaluate
    pin would leave 0.0."""
    registered(_OneRowScheme)
    mesh = _mesh(bodies=_cylinder())
    T = _quadratic_field(mesh)
    eqn = Equation(exp.laplacian(1.0, T))
    evaluate(eqn, t=0.0, solution=_sol(METHOD))

    _seed_solid(T, 7.0)
    evaluate(eqn, t=0.0, solution=_sol(METHOD))

    np.testing.assert_array_equal(_valid_cells(T)[..., 0][_solid_columns()], 7.0)


def test_the_pin_does_not_run_again_in_a_later_solve(blockamr_session, registered):
    """The same once-per-key observable at the ``solve()`` seam (review.md §4
    Q19, the gap Q12 left open when B25 proved the property through
    ``evaluate`` only). ``solve()`` builds its own ``BandEvaluation`` per call,
    whose construction is where the pin lives — so "one driver per call" and
    "one pin per key" are genuinely different claims, and only this test rules
    out the first implying the second.

    ``dt = 0.0`` makes the Euler update the identity (``T += 0 * src``), so the
    pin is the *only* thing in a ``solve()`` that can write a solid cell and the
    probe stays bitwise, exactly like its ``evaluate`` twin. Seeding 7.0 into
    the solid columns between the two solves and finding it there afterwards is
    the whole claim; a per-call pin would leave 0.0."""
    registered(_OneRowScheme)
    mesh = _mesh(bodies=_cylinder())
    T = _quadratic_field(mesh)
    eqn = Equation(exp.ddt(T) - exp.laplacian(1.0, T), schemes={"ddt": "Euler"})
    solve(eqn, dt=0.0, t=0.0, solution=_sol(METHOD))

    _seed_solid(T, 7.0)
    solve(eqn, dt=0.0, t=0.0, solution=_sol(METHOD))

    np.testing.assert_array_equal(_valid_cells(T)[..., 0][_solid_columns()], 7.0)


def test_a_new_generation_pins_the_field_again(blockamr_session, registered):
    """The memo is keyed on ``grid_version`` like every other IBM cache (design
    §8), so a regrid, a moved body or an explicit ``invalidate()`` re-pins: the
    classification the pin belongs to has been redone."""
    registered(_OneRowScheme)
    mesh = _mesh(bodies=_cylinder())
    T = _quadratic_field(mesh)
    eqn = Equation(exp.laplacian(1.0, T))
    evaluate(eqn, t=0.0, solution=_sol(METHOD))
    _seed_solid(T, 7.0)

    mesh.ibm.invalidate()
    evaluate(eqn, t=0.0, solution=_sol(METHOD))

    np.testing.assert_array_equal(_valid_cells(T)[..., 0][_solid_columns()], 0.0)


def test_a_second_field_is_pinned_on_its_own_first_evaluate(blockamr_session, registered):
    """The **field** is part of the key, on top of the design's
    ``(method, lev, grid_version)`` triple (review.md §4 Q3): the write lands in
    that field's storage, so a field created after the classification — same
    mesh, same generation — is still pinned the first time it is evaluated."""
    registered(_OneRowScheme)
    mesh = _mesh(bodies=_cylinder())
    first = _quadratic_field(mesh)
    evaluate(Equation(exp.laplacian(1.0, first)), t=0.0, solution=_sol(METHOD))

    second = _quadratic_field(mesh)
    _seed_solid(second, 7.0)
    evaluate(Equation(exp.laplacian(1.0, second)), t=0.0, solution=_sol(METHOD))

    np.testing.assert_array_equal(_valid_cells(second)[..., 0][_solid_columns()], 0.0)
