// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// Compile-time proof that IsMatrix and IsOperator are satisfiable, that they name the
// members they are meant to name, and that they DISCRIMINATE -- a concept that
// accidentally admitted every type would satisfy the first two and be worthless.
// Nothing here is instantiated, linked or called -- the translation unit exists so that
// a mistake in coefficients.hpp is a compile error here, beside the concept, rather than
// at the first real format.
//
// It lives in the shipped object library rather than under test/ because blockAmr has
// no C++ test target (NeoN_BUILD_TESTS is OFF for the Python build and test/CMakeLists
// does not add a blockAmr subdirectory), so a test TU would never be compiled at all.
// The real formats' own static_asserts (faceCoeffMatrix.hpp) reach a compiler for the
// same reason: this TU includes them.

#include <optional>
#include <type_traits>

#include "NeoN/blockAmr/linearAlgebra/coefficients.hpp"
#include "NeoN/blockAmr/linearAlgebra/faceCoeffMatrix.hpp"
#include "NeoN/blockAmr/linearAlgebra/linearSystem.hpp"
#include "NeoN/blockAmr/linearAlgebra/matrix.hpp"
#include "NeoN/blockAmr/linearAlgebra/operator.hpp"
#include "NeoN/blockAmr/operators/laplacian.hpp"

// A named namespace, not an anonymous one: with internal linkage nvcc reports every
// member as "declared but never referenced" (#177-D), which a concept check never is.
// Nothing is defined here, so no symbol is emitted either way.
namespace blockamr::la::conceptCheck
{

// Declared, not defined: a concept checks the signatures in an unevaluated context, so
// bodies would only add unreferenced-function warnings.
struct StubMatrix
{
    std::shared_ptr<const gko::LinOp> op() const;
    bool isAssembled() const;
    MatrixCoefficients coefficients();
    void zero();
    Symmetry symmetry() const;
    std::size_t localRows() const;
    const NeoN::Executor& executor() const;
    std::shared_ptr<const gko::LinOp> makePrecond(const SolverConfig&) const;
    const char* name() const;
};

struct StubOperator
{
    void assemble(Coefficients) const;
};

static_assert(IsMatrix<StubMatrix>);
static_assert(IsOperator<StubOperator>);

// --- Negative side: the concepts REJECT, they do not merely accept ---------
//
// Each stub below is StubMatrix/StubOperator with exactly ONE thing wrong, so a
// failure names the requirement that stopped discriminating. Without these the TU
// would pass unchanged if `requires` had been mistyped into something vacuously
// true (an empty requires-block, a misplaced `;`, a requires-expression whose
// body is never checked) -- the classic way a concept quietly admits everything.

// Missing member: zero() is gone, nothing else changed.
struct StubMatrixNoZero
{
    std::shared_ptr<const gko::LinOp> op() const;
    bool isAssembled() const;
    MatrixCoefficients coefficients();
    Symmetry symmetry() const;
    std::size_t localRows() const;
    const NeoN::Executor& executor() const;
    std::shared_ptr<const gko::LinOp> makePrecond(const SolverConfig&) const;
    const char* name() const;
};

// Present member, WRONG return type: localRows() returns int, not std::size_t.
// This is what pins the `-> std::same_as<...>` half of each requirement; a
// concept checking only that the expression is valid would accept this one.
struct StubMatrixNarrowLocalRows
{
    std::shared_ptr<const gko::LinOp> op() const;
    bool isAssembled() const;
    MatrixCoefficients coefficients();
    void zero();
    Symmetry symmetry() const;
    int localRows() const;
    const NeoN::Executor& executor() const;
    std::shared_ptr<const gko::LinOp> makePrecond(const SolverConfig&) const;
    const char* name() const;
};

// Present member, WRONG parameter: makePrecond takes a PrecondKind rather than the
// whole SolverConfig. A precond kind alone cannot shape a V-cycle -- the sweep
// counts, omega and precision live on the config -- so this is the mistake the
// signature is most likely to drift into, and it is pinned rather than described.
struct StubMatrixPrecondKindOnly
{
    std::shared_ptr<const gko::LinOp> op() const;
    bool isAssembled() const;
    MatrixCoefficients coefficients();
    void zero();
    Symmetry symmetry() const;
    std::size_t localRows() const;
    const NeoN::Executor& executor() const;
    std::shared_ptr<const gko::LinOp> makePrecond(PrecondKind) const;
    const char* name() const;
};

// Present member, WRONG parameter: assemble() takes MatrixCoefficients (the
// matrix-side subset) rather than the Coefficients an operator is handed, which
// carries the rhs as well. Coefficients is not convertible to MatrixCoefficients,
// so the call in the requires-expression does not compile and the concept fails.
struct StubOperatorWrongArgument
{
    void assemble(MatrixCoefficients) const;
};

static_assert(!IsMatrix<StubMatrixNoZero>);
static_assert(!IsMatrix<StubMatrixNarrowLocalRows>);
static_assert(!IsMatrix<StubMatrixPrecondKindOnly>);
static_assert(!IsOperator<StubOperatorWrongArgument>);

// A type that is not a matrix at all -- the floor these concepts must clear.
static_assert(!IsMatrix<int>);
static_assert(!IsOperator<int>);

// --- The S5 gate, asserted rather than described ---------------------------
//
// Two privacy rules carry the claim that an operator can only run through
// `system += op`. Neither is testable from Python -- the whole point is that the
// code that would test them does not compile -- so they are asserted here.

// 1. Operator::assemble is PRIVATE (friend: LinearSystem). The erasure therefore
//    does NOT satisfy IsOperator, which is the one place its shape deliberately
//    departs from Matrix (matrix.hpp asserts IsMatrix<Matrix> to check its own
//    forwarding surface; Operator cannot, and must not). Make `assemble` public
//    and this fires.
static_assert(!IsOperator<Operator>);

// 2. Coefficients has a private constructor (friend: LinearSystem). A concrete
//    operator's assemble() is public -- IsOperator requires it -- so the argument
//    is what is actually out of reach: there is no way to PRODUCE one except
//    through operator+=. std::is_constructible respects access control, so this
//    is the check, not a comment about the check.
static_assert(!std::is_constructible_v<Coefficients, MatrixCoefficients, CellFieldLevel>);
// Not copy-constructible from thin air either -- but a Coefficients a caller was
// handed can of course be copied, which is what makes assemble(Coefficients) a
// by-value parameter rather than a reference.
static_assert(std::is_copy_constructible_v<Coefficients>);

// --- Absence has exactly ONE spelling, and only `lower` has it ----------------
//
// `diag` and `upper` are never absent. Asserted here rather than probed from
// Python: with CellFieldLevel/FaceFieldLevel non-nullable, "empty" is not
// expressible at all, so this is what _la_matrix_probe's diag_empty/upper_empty
// keys became -- they now emit a literal false, and these three lines are what
// makes that literal true. Turn either member into an optional, or take the
// optional off `lower`, and this fires.
static_assert(std::is_same_v<decltype(MatrixCoefficients::diag), CellFieldLevel>);
static_assert(std::is_same_v<decltype(MatrixCoefficients::upper), FaceFieldLevel>);
static_assert(std::is_same_v<decltype(MatrixCoefficients::lower), std::optional<FaceFieldLevel>>);
static_assert(std::is_same_v<decltype(Coefficients::diag), CellFieldLevel>);
static_assert(std::is_same_v<decltype(Coefficients::upper), FaceFieldLevel>);
static_assert(std::is_same_v<decltype(Coefficients::lower), std::optional<FaceFieldLevel>>);
static_assert(std::is_same_v<decltype(Coefficients::rhs), CellFieldLevel>);

// --- The coefficients are SELF-DESCRIBING: the mesh travels with them ---------
//
// Not optional and not a pointer. An operator cannot write a face coefficient
// without dx, so a coefficient set that does not carry its own layout is not
// sufficient to assemble from -- and passing it alongside instead made a mismatch
// representable (ops::Laplacian used to hold its own amrex::Geometry). Take this
// member off either struct and every operator needs a second source for it again.
static_assert(std::is_same_v<decltype(MatrixCoefficients::mesh), MeshLevel>);
static_assert(std::is_same_v<decltype(Coefficients::mesh), MeshLevel>);

// The geometry is NOT separately constructible into ops::Laplacian any more: the
// three-argument form that took one is gone, so the old
// `Laplacian(gamma, geom, bc)` spelling no longer compiles.
static_assert(!std::is_constructible_v<
              ops::Laplacian,
              const amrex::MultiFab&,
              amrex::Geometry,
              BcArray>);

// The concrete operator satisfies the concept it is written against. (It also
// asserts this in its own header; repeated here so this TU fails if the header's
// assertion is ever removed along with whatever broke it.)
static_assert(IsOperator<ops::Laplacian>);

} // namespace blockamr::la::conceptCheck
