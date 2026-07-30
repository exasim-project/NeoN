// SPDX-FileCopyrightText: 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ginkgo/ginkgo.hpp>

#include <concepts>
#include <cstddef>
#include <memory>
#include <optional>
#include <utility>

#include "NeoN/blockAmr/core/bc.hpp"
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

class LinearSystem;

// What an OPERATOR writes through: the matrix coefficients plus the rhs, which lives on the
// system rather than the matrix. Format-agnostic. Only LinearSystem can build one (private
// ctor + friend), so `system += op` is the only route to a Coefficients -- and its ctor is
// the single site where the six fields get ORDERED, where a transposition would otherwise hide.
class Coefficients
{
public:

    // Travels with the coefficients because a face coefficient cannot be written without dx
    // and the periodicity.
    MeshLevel mesh;
    // The cell-centred diagonal SOURCE (ddt/Sp/reaction), NOT the matrix diagonal
    // alpha - (aE+aW+aN+aS+aT+aB), which the format derives (faceCoeffMatrix.hpp).
    CellFieldLevel alpha;
    FaceFieldLevel upper;
    // No low side to WRITE when symmetric: an operator writing both would double every
    // coefficient. Storage still has one -- the format's storedLower().
    std::optional<FaceFieldLevel> lower; // nullopt when symmetric
    CellFieldLevel rhs;
    // Where the operator's kernels launch (blockamr::parallelFor). It comes from the MATRIX,
    // the only thing that knows where the coefficient FIELDS live.
    NeoN::Executor exec {NeoN::SerialExecutor {}};

private:

    friend class LinearSystem;

    Coefficients(
        MeshLevel mesh,
        CellFieldLevel alpha,
        FaceFieldLevel upper,
        std::optional<FaceFieldLevel> lower,
        CellFieldLevel rhs,
        const NeoN::Executor& exec
    )
        : mesh(std::move(mesh)), alpha(alpha), upper(upper), lower(lower), rhs(rhs), exec(exec)
    {}
};

// What it takes to BE a matrix format -- no base class. The coefficient fields are required as
// MEMBERS: they are public in every format (writing all six is an operator's whole job), and a
// data member cannot share a name with a member function, so there are no accessors to require.
// makePrecond is the FORMAT'S job: a solver holds only a gko::LinOp, by which point erasure has
// discarded the coefficients a GMG hierarchy rediscretises. Null = declined.
// report/blockamr-linear-algebra-notes.md
template<typename T>
concept IsMatrix = requires(T t, const T ct, const SolverConfig& config) {
    {
        t.exec
    } -> std::same_as<NeoN::Executor&>;
    {
        t.bc
    } -> std::same_as<BcArray&>;
    {
        t.mesh
    } -> std::same_as<MeshLevel&>;
    {
        t.alpha
    } -> std::same_as<CellFieldLevel&>;
    {
        t.upper
    } -> std::same_as<FaceFieldLevel&>;
    {
        t.lower
    } -> std::same_as<std::optional<FaceFieldLevel>&>;
    {
        ct.op()
    } -> std::same_as<std::shared_ptr<const gko::LinOp>>;
    {
        ct.isAssembled()
    } -> std::same_as<bool>;
    // Whatever the format DERIVES from the coefficients (an assembly, a stored diagonal) is
    // stale once they are written, and there is no "done writing" call to hook -- so acquiring
    // a write handle marks it through this, pessimistically.
    {
        t.markStale()
    } -> std::same_as<void>;
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
