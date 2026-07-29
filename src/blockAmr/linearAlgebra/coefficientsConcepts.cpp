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
static_assert(!std::is_constructible_v<Coefficients, MatrixCoefficients, CellView>);
// Not copy-constructible from thin air either -- but a Coefficients a caller was
// handed can of course be copied, which is what makes assemble(Coefficients) a
// by-value parameter rather than a reference.
static_assert(std::is_copy_constructible_v<Coefficients>);

// The concrete operator satisfies the concept it is written against. (It also
// asserts this in its own header; repeated here so this TU fails if the header's
// assertion is ever removed along with whatever broke it.)
static_assert(IsOperator<ops::Laplacian>);

} // namespace blockamr::la::conceptCheck
