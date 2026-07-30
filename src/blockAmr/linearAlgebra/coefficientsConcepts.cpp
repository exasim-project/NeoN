// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

// Compile-time proof that IsMatrix and IsOperator are satisfiable, name the members they mean
// to, and DISCRIMINATE. In the shipped object library, not under test/, because blockAmr has no
// C++ test target -- and the formats' own static_asserts reach a compiler only through this TU.

#include <optional>
#include <type_traits>

#include "NeoN/blockAmr/linearAlgebra/coefficients.hpp"
#include "NeoN/blockAmr/linearAlgebra/faceCoeffMatrix.hpp"
#include "NeoN/blockAmr/linearAlgebra/linearSystem.hpp"
#include "NeoN/blockAmr/linearAlgebra/matrix.hpp"
#include "NeoN/blockAmr/linearAlgebra/operator.hpp"
#include "NeoN/blockAmr/operators/laplacian.hpp"

// Named, not anonymous: with internal linkage nvcc reports every member as "declared but never
// referenced" (#177-D), which a concept check never is.
namespace blockamr::la::conceptCheck
{

// The member functions are declared, not defined: a concept checks signatures in an unevaluated
// context. The coefficient fields have to be real members, since that is what IsMatrix requires.
struct StubMatrix
{
    NeoN::Executor exec;
    BcArray bc;
    MeshLevel mesh;
    CellFieldLevel alpha;
    FaceFieldLevel upper;
    std::optional<FaceFieldLevel> lower;

    std::shared_ptr<const gko::LinOp> op() const;
    bool isAssembled() const;
    void markStale();
    void zero();
    Symmetry symmetry() const;
    std::size_t localRows() const;
    std::shared_ptr<const gko::LinOp> makePrecond(const SolverConfig&) const;
    const char* name() const;
};

struct StubOperator
{
    void assemble(Coefficients) const;
};

static_assert(IsMatrix<StubMatrix>);
static_assert(IsOperator<StubOperator>);

// Each stub below is StubMatrix/StubOperator with exactly ONE thing wrong, so a failure names
// the requirement that stopped discriminating -- and a `requires` mistyped into something
// vacuously true cannot pass them.

// Missing member: zero() is gone, nothing else changed.
struct StubMatrixNoZero
{
    NeoN::Executor exec;
    BcArray bc;
    MeshLevel mesh;
    CellFieldLevel alpha;
    FaceFieldLevel upper;
    std::optional<FaceFieldLevel> lower;

    std::shared_ptr<const gko::LinOp> op() const;
    bool isAssembled() const;
    void markStale();
    Symmetry symmetry() const;
    std::size_t localRows() const;
    std::shared_ptr<const gko::LinOp> makePrecond(const SolverConfig&) const;
    const char* name() const;
};

// Present member, WRONG return type: pins the `-> std::same_as<...>` half of each requirement,
// which a validity-only concept would accept.
struct StubMatrixNarrowLocalRows
{
    NeoN::Executor exec;
    BcArray bc;
    MeshLevel mesh;
    CellFieldLevel alpha;
    FaceFieldLevel upper;
    std::optional<FaceFieldLevel> lower;

    std::shared_ptr<const gko::LinOp> op() const;
    bool isAssembled() const;
    void markStale();
    void zero();
    Symmetry symmetry() const;
    int localRows() const;
    std::shared_ptr<const gko::LinOp> makePrecond(const SolverConfig&) const;
    const char* name() const;
};

// Present member, WRONG parameter: a PrecondKind alone cannot shape a V-cycle (sweeps, omega
// and precision live on the SolverConfig), so this is the likeliest drift -- pinned, not
// described.
struct StubMatrixPrecondKindOnly
{
    NeoN::Executor exec;
    BcArray bc;
    MeshLevel mesh;
    CellFieldLevel alpha;
    FaceFieldLevel upper;
    std::optional<FaceFieldLevel> lower;

    std::shared_ptr<const gko::LinOp> op() const;
    bool isAssembled() const;
    void markStale();
    void zero();
    Symmetry symmetry() const;
    std::size_t localRows() const;
    std::shared_ptr<const gko::LinOp> makePrecond(PrecondKind) const;
    const char* name() const;
};

// Present field, WRONG constness: the coefficient requirements read
// `{ t.alpha } -> std::same_as<CellFieldLevel&>`, so a const member yields a const& and stops
// matching. This is the drift the MEMBER requirements are exposed to and the accessors were
// not, and a requires-expression that only checked validity would accept it.
struct StubMatrixConstAlpha
{
    NeoN::Executor exec;
    BcArray bc;
    MeshLevel mesh;
    const CellFieldLevel alpha;
    FaceFieldLevel upper;
    std::optional<FaceFieldLevel> lower;

    std::shared_ptr<const gko::LinOp> op() const;
    bool isAssembled() const;
    void markStale();
    void zero();
    Symmetry symmetry() const;
    std::size_t localRows() const;
    std::shared_ptr<const gko::LinOp> makePrecond(const SolverConfig&) const;
    const char* name() const;
};

// Present member, WRONG parameter: assemble() takes one coefficient field rather than the
// Coefficients an operator is handed, which carries the mesh, the faces and the rhs too. The
// two are not convertible, so the call in the requires-expression does not compile.
struct StubOperatorWrongArgument
{
    void assemble(CellFieldLevel) const;
};

static_assert(!IsMatrix<StubMatrixNoZero>);
static_assert(!IsMatrix<StubMatrixNarrowLocalRows>);
static_assert(!IsMatrix<StubMatrixPrecondKindOnly>);
static_assert(!IsMatrix<StubMatrixConstAlpha>);
static_assert(!IsOperator<StubOperatorWrongArgument>);

// A type that is not a matrix at all -- the floor these concepts must clear.
static_assert(!IsMatrix<int>);
static_assert(!IsOperator<int>);

// Two privacy rules carry the claim that an operator can only run through `system += op`.
// Neither is testable from Python -- the code that would test them does not compile -- so they
// are asserted here.

// 1. Operator::assemble is PRIVATE (friend: LinearSystem), so the erasure does NOT satisfy
//    IsOperator -- for a reason of its own, not the reason Matrix fails IsMatrix (which wants
//    the coefficient DATA MEMBERS). Make `assemble` public and this fires.
static_assert(!IsOperator<Operator>);

// 2. Coefficients' ctor is private (friend: LinearSystem) and is_constructible respects access
//    control, so operator+= is the only way to PRODUCE one. The list is the REAL 6-arg ctor's;
//    StubPrivateCtor asserts it BINDS, without which the negative below would go vacuous.
struct StubPrivateCtor
{
    StubPrivateCtor(
        MeshLevel mesh,
        CellFieldLevel alpha,
        FaceFieldLevel upper,
        std::optional<FaceFieldLevel> lower,
        CellFieldLevel rhs,
        const NeoN::Executor& exec
    );
};
static_assert(std::is_constructible_v<
              StubPrivateCtor,
              MeshLevel,
              CellFieldLevel,
              FaceFieldLevel,
              std::optional<FaceFieldLevel>,
              CellFieldLevel,
              NeoN::Executor>);
static_assert(!std::is_constructible_v<
              Coefficients,
              MeshLevel,
              CellFieldLevel,
              FaceFieldLevel,
              std::optional<FaceFieldLevel>,
              CellFieldLevel,
              NeoN::Executor>);
// A Coefficients a caller was handed can of course be copied -- which is what makes
// assemble(Coefficients) a by-value parameter rather than a reference.
static_assert(std::is_copy_constructible_v<Coefficients>);

// Absence has exactly ONE spelling and only `lower` has it: with CellFieldLevel/FaceFieldLevel
// non-nullable, "empty" is not expressible, which is why _la_matrix_probe's diag_empty/
// upper_empty keys now emit a literal false. Turn either into an optional and this fires. The
// FORMATS' own fields are pinned by IsMatrix's member requirements instead, which
// static_assert(IsMatrix<MFFaceCoeffs>) / <CsrMatrix> check for both of them.
static_assert(std::is_same_v<decltype(Coefficients::alpha), CellFieldLevel>);
static_assert(std::is_same_v<decltype(Coefficients::upper), FaceFieldLevel>);
static_assert(std::is_same_v<decltype(Coefficients::lower), std::optional<FaceFieldLevel>>);
static_assert(std::is_same_v<decltype(Coefficients::rhs), CellFieldLevel>);

// The mesh TRAVELS with the coefficients, not optional and not a pointer: an operator cannot
// write a face coefficient without dx, and passing it alongside made a mismatch representable
// (ops::Laplacian used to hold its own amrex::Geometry).
static_assert(std::is_same_v<decltype(Coefficients::mesh), MeshLevel>);

// ops::Laplacian takes no Geometry any more: the old `Laplacian(gamma, geom, bc)` spelling
// does not compile.
static_assert(!std::is_constructible_v<
              ops::Laplacian,
              const amrex::MultiFab&,
              amrex::Geometry,
              BcArray>);

// Repeated from the operator's own header, so this TU fails if that assertion is removed.
static_assert(IsOperator<ops::Laplacian>);

} // namespace blockamr::la::conceptCheck
