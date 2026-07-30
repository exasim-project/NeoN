// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ginkgo/ginkgo.hpp>

#include <concepts>
#include <cstddef>
#include <memory>
#include <optional>

#include "NeoN/blockAmr/core/fieldLevel.hpp"
#include "NeoN/blockAmr/core/meshLevel.hpp"
#include "NeoN/blockAmr/linearAlgebra/solverConfig.hpp"
#include "NeoN/core/executor/executor.hpp"

namespace blockamr::la
{

// The coefficient handles are deliberately NARROW (core/fieldLevel.hpp) and stand for BOTH
// formats: assembled and matrix-free storage is MultiFab-shaped either way, so no type
// erasure lives here. Why, and when to revisit: report/blockamr-linear-algebra-notes.md

enum class Symmetry
{
    symmetric,
    asymmetric
};

// The matrix-side handles, grouped by role -- lduMatrix's split. `diag` is STILL the
// diagonal SOURCE alpha, not the matrix diagonal (faceCoeffMatrix.hpp); `mesh` travels with
// them because a face coefficient cannot be written without dx and the periodicity.
struct MatrixCoefficients
{
    MeshLevel mesh;
    CellFieldLevel diag;
    FaceFieldLevel upper;
    // No low side to WRITE when symmetric: an operator writing both would double every
    // coefficient. Storage still has one -- FaceCoeffFields::storedLower().
    std::optional<FaceFieldLevel> lower; // nullopt when symmetric

    bool symmetric() const { return !lower.has_value(); }
};

class LinearSystem;

// What an OPERATOR writes through: the matrix coefficients plus the rhs, which lives on the
// system rather than the matrix. Format-agnostic. Only LinearSystem can build one (private
// ctor + friend), so `system += op` is the only route to a Coefficients.
class Coefficients
{
public:

    MeshLevel mesh;
    CellFieldLevel diag;
    FaceFieldLevel upper;
    std::optional<FaceFieldLevel> lower; // nullopt when symmetric
    CellFieldLevel rhs;
    // Where the operator's kernels launch (blockamr::parallelFor). It comes from the MATRIX,
    // the only thing that knows where the coefficient FIELDS live.
    NeoN::Executor exec {NeoN::SerialExecutor {}};

private:

    friend class LinearSystem;

    Coefficients(MatrixCoefficients mc, CellFieldLevel rhs, const NeoN::Executor& exec)
        : mesh(mc.mesh), diag(mc.diag), upper(mc.upper), lower(mc.lower), rhs(rhs), exec(exec)
    {}
};

// What it takes to BE a matrix format -- no base class. makePrecond is the FORMAT'S job: a
// solver holds only a gko::LinOp, by which point erasure has discarded the coefficients a
// GMG hierarchy rediscretises. Null = declined. report/blockamr-linear-algebra-notes.md
template<typename T>
concept IsMatrix = requires(T t, const T ct, const SolverConfig& config) {
    {
        ct.op()
    } -> std::same_as<std::shared_ptr<const gko::LinOp>>;
    {
        ct.isAssembled()
    } -> std::same_as<bool>;
    {
        t.coefficients()
    } -> std::same_as<MatrixCoefficients>;
    {
        t.zero()
    } -> std::same_as<void>;
    {
        ct.symmetry()
    } -> std::same_as<Symmetry>;
    {
        ct.localRows()
    } -> std::same_as<std::size_t>;
    {
        ct.executor()
    } -> std::same_as<const NeoN::Executor&>;
    {
        ct.makePrecond(config)
    } -> std::same_as<std::shared_ptr<const gko::LinOp>>;
    {
        ct.name()
    } -> std::same_as<const char*>;
};

// What it takes to BE an operator. Declared HERE so linearAlgebra/ never depends on the
// operators -- the dependency runs ops -> la.
template<typename T>
concept IsOperator = requires(const T t, Coefficients c) {
    {
        t.assemble(c)
    } -> std::same_as<void>;
};

} // namespace blockamr::la
