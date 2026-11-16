// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

// included by the generated umbrella header NeoN/NeoN.hpp regardless of the build options
#ifdef NeoN_WITH_JULIA

#include <chrono>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <variant>

#include <julia.h>

#include "NeoN/core/dictionary.hpp"
#include "NeoN/core/error.hpp"
#include "NeoN/core/logging.hpp"
#include "NeoN/core/database/oldTimeCollection.hpp"
#include "NeoN/dsl/operator.hpp"
#include "NeoN/linearAlgebra/linearSystem.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/stencil/geometryScheme.hpp"

namespace NeoN::dsl
{

template<typename ValueType, typename IndexType>
class Expression;

/* @brief Matrix assembly through the JuNe Julia package.
 *
 * Julia must be initialised (jl_init + include(JUNE_MODULE_INIT)) by the executable, and every
 * call must come from the thread that ran jl_init.
 */
namespace june
{

enum class Strategy
{
    FromIterator, // CellBasedIterator -> Cell, BatchedFaceIterator -> Batched,
                  // GlobalFaceBasedIterator -> GlobalFace, else -> Face
    Face,
    GlobalFace,
    Cell,
    Batched, // face-based batch by batch (la::BatchedFaceIterator), no atomics on internal faces
    BatchedCUDA // Batched with JuNe's CUDA.jl kernels instead of KernelAbstractions (GPU only)
};

/* @brief whether JuNe implements every implicit operator of the expression */
/* @brief Whether JuNe takes over Expression::assemble: initially when NeoN_JULIA_ASSEMBLY was
 * compiled in or the environment variable NEON_JUNE_ASSEMBLY is set to anything but 0, and
 * switchable at run time with setEnabled(). */
inline bool& enabledFlag()
{
    static bool on = []
    {
#ifdef NeoN_JULIA_ASSEMBLY
        return true;
#else
        const char* value = std::getenv("NEON_JUNE_ASSEMBLY");
        return value != nullptr && *value != '\0' && std::string_view(value) != "0";
#endif
    }();
    return on;
}

inline bool enabled() { return enabledFlag(); }

inline void setEnabled(bool on) { enabledFlag() = on; }

template<typename ValueType, typename IndexType>
bool canAssemble(const Expression<ValueType, IndexType>& expr)
{
    bool hasSpatial = false;
    for (const auto& op : expr.spatialOperators())
    {
        if (op.getType() != Operator::Type::Implicit) continue;
        const auto name = op.getName();
        const auto jlOp = op.juliaOP();
        // JuNe only defines the upwind and linear div schemes; its linearUpwind assembles the
        // upwind implicit part without NeoN's explicit gradient correction of the rhs
        const bool supportedDiv = name == "DivOperator"
                               && (jlOp.starts_with("Div{Float64, upwind{")
                                   || jlOp.starts_with("Div{Float64, linear{"));
        if (name != "LaplacianOperator" && !supportedDiv) return false;
        hasSpatial = true;
    }
    for (const auto& op : expr.temporalOperators())
    {
        if (op.getType() != Operator::Type::Implicit) continue;
        // an empty string is steadyState, which assembles nothing; JuNe rewrites BDF2 to BDF1
        const auto jlOp = op.juliaOP();
        if (op.getName() != "DdtOperator" || jlOp.find("BDF2") != std::string::npos) return false;
    }
    // phi, and with it the mesh, comes from the spatial operators
    return hasSpatial;
}

/* @brief the Julia expression of all implicit operators, built from the current scheme reads */
template<typename ValueType, typename IndexType>
std::string opString(const Expression<ValueType, IndexType>& expr)
{
    std::string ret;
    auto append = [&ret](const auto& op)
    {
        if (op.getType() != Operator::Type::Implicit || op.juliaOP().empty()) return;
        ret += ret.empty() ? op.juliaOP() : " + " + op.juliaOP();
    };
    for (const auto& op : expr.temporalOperators())
    {
        append(op);
    }
    for (const auto& op : expr.spatialOperators())
    {
        append(op);
    }
    return ret;
}

namespace detail
{

inline void checkJuliaException()
{
    if (jl_exception_occurred())
    {
        const char* p =
            jl_string_ptr(jl_eval_string("sprint(showerror, ccall(:jl_exception_occurred, Any, ()))"
            ));
        NF_ERROR_EXIT(std::string("JuNe assembly failed: ") + p);
    }
}

/* @brief returns the config entry of the first implicit spatial operator named opName */
template<typename HoldType, typename ValueType, typename IndexType>
const HoldType* findConfig(
    const Expression<ValueType, IndexType>& expr, const std::string& opName, const std::string& key
)
{
    for (const auto& op : expr.spatialOperators())
    {
        if (op.getType() == Operator::Type::Implicit && op.getName() == opName)
        {
            return &op.getConfig().template get<NeoN::detail::RefHolder<HoldType>>(key).c;
        }
    }
    return nullptr;
}

} // namespace detail

/* @brief assemble all implicit operators of expr into ls with JuNe
 *
 * Only valid if canAssemble(expr). JuNe adds to the existing coefficients, so ls must be reset.
 */
template<typename ValueType, typename IndexType, typename MatrixValueType = ValueType>
void assemble(
    const Expression<ValueType, IndexType>& expr,
    la::LinearSystem<MatrixValueType, ValueType>& ls,
    scalar dt,
    Strategy strategy = Strategy::FromIterator
)
{
    namespace fvcc = NeoN::finiteVolume::cellCentred;

    const auto* divPhi =
        detail::findConfig<fvcc::VolumeField<ValueType>>(expr, "DivOperator", "field");
    const auto* lapPhi =
        detail::findConfig<fvcc::VolumeField<ValueType>>(expr, "LaplacianOperator", "field");
    NF_ASSERT(divPhi || lapPhi, "JuNe assembly needs a div or laplacian operator");
    const auto& phi = divPhi ? *divPhi : *lapPhi;
    const auto& mesh = phi.mesh();

    // JuNe skips the inputs of operators that are not in the op string, so absent operators
    // get empty arrays
    const auto empty = Vector<scalar>(ls.exec(), 0);
    const auto emptyOld = Vector<ValueType>(ls.exec(), 0);
    const auto* faceFlux =
        detail::findConfig<fvcc::SurfaceField<scalar>>(expr, "DivOperator", "flux");
    const auto* gamma =
        detail::findConfig<fvcc::SurfaceField<scalar>>(expr, "LaplacianOperator", "gamma");
    const auto fusedOPString = opString(expr);
    const bool hasDdt = fusedOPString.starts_with("Ddt{");

    // The GC does not scan C++ locals: every Julia object created below (array wrappers, boxes,
    // the VectorType objects) is rooted in this frame until the call, else the allocations of the
    // later ones can collect the earlier ones (Julia embedding docs, "Memory Management").
    jl_value_t** roots;
    JL_GC_PUSHARGS(roots, 48);
    size_t nRoots = 0;
    auto root = [&](auto* value)
    {
        NF_ASSERT(nRoots < 48, "JuNe assembly: GC root frame too small");
        roots[nRoots++] = (jl_value_t*)value;
        return value;
    };
    try
    {

    const auto ifaceFluxV = root(faceFlux ? faceFlux->internalVector().juliaPtr() : empty.juliaPtr());
    const auto bfaceFluxV = root(faceFlux ? faceFlux->boundaryData().value().juliaPtr() : empty.juliaPtr());
    const auto iGamma = root(gamma ? gamma->internalVector().juliaPtr() : empty.juliaPtr());
    const auto bGamma = root(gamma ? gamma->boundaryData().value().juliaPtr() : empty.juliaPtr());
    const auto oldVector = root(hasDdt ? fvcc::oldTime(phi).internalVector().juliaPtr() : emptyOld.juliaPtr());

    const auto nInternalFaces = mesh.nInternalFaces();
    const auto nTotalFaces = mesh.nTotalFaces();
    const auto nCells = mesh.nCells();
    bool useGPU = std::holds_alternative<GPUExecutor>(mesh.exec());
    // NeoN's pending kernels, e.g. the reset of the system, run on Kokkos' stream, which the
    // CUDA.jl stream of the JuNe kernels does not wait for
    fence(mesh.exec());

    // the cached geometry-scheme coefficients: the mesh centres they are computed from are
    // released after the first geometry read
    const auto& nonOrthDeltaCoeffs = fvcc::GeometryScheme::readOrCreate(mesh)->nonOrthDeltaCoeffs();
    const auto ideltaCoeffs = root(nonOrthDeltaCoeffs.internalVector().juliaPtr());
    const auto bdeltaCoeffs = root(nonOrthDeltaCoeffs.boundaryData().value().juliaPtr());

    const auto magFaceAreas = root(mesh.faceAreas().juliaPtr());
    const auto boundaryFaceOwners = root(mesh.boundaryMesh().faceOwners().juliaPtr());
    const auto vol = root(mesh.cellVolumes().juliaPtr());
    auto refGrad = root(phi.boundaryData().refGrad().juliaPtr());
    auto valueFraction = root(phi.boundaryData().valueFraction().juliaPtr());
    auto refValue = root(phi.boundaryData().refValue().juliaPtr());
    auto rhsPtr = root(ls.rhs().juliaPtr());
    auto bRhs = root(ls.boundaryRhs().juliaPtr());

    auto iterator = std::dynamic_pointer_cast<la::CellBasedIterator>(ls.getMeshIterator()->get());
    auto batchedIterator =
        std::dynamic_pointer_cast<la::BatchedFaceIterator>(ls.getMeshIterator()->get());
    if (strategy == Strategy::FromIterator)
    {
        const bool globalIterator =
            std::dynamic_pointer_cast<la::GlobalFaceBasedIterator>(ls.getMeshIterator()->get())
            != nullptr;
        strategy = iterator          ? Strategy::Cell
                 : batchedIterator   ? Strategy::Batched
                 : globalIterator    ? Strategy::GlobalFace
                                     : Strategy::Face;
    }

    jl_module_t* mod = (jl_module_t*)jl_eval_string("JuNe");
    jl_value_t* vectorType = root(jl_get_global(
        mod,
        jl_symbol("VectorType")
    ));
    // the matrix value type: NeoN's coupled Vec3 matrix or the segregated scalar one
    jl_value_t* vectorValueType = root(jl_get_global(
        mod,
        jl_symbol(std::is_same_v<MatrixValueType, NeoN::Vec3> ? "Vec3" : "Scalar")
    ));
    jl_value_t* params[] = { vectorValueType };
    jl_value_t* concrete_type = root(jl_apply_type(
        vectorType,
        params,
        1
    ));

    
    if (strategy == Strategy::Cell)
    {
        NF_ASSERT(iterator, "JuNe cell-based assembly needs a CellBasedIterator");
        const auto matIt = ls.faceToMatrixAddress();
        if (!iterator->getCellBasedData())
        {
            iterator->setComputeCellBasedData(mesh, ls.matrix().sparsity(), matIt);
        }
        auto cellBasedData = iterator->getCellBasedData();

        auto [cellFacesValues, cellFacesSegments] = cellBasedData->cellFaces.juliaPtrs();
        root(cellFacesValues);
        root(cellFacesSegments);
        auto faceSignV = root(cellBasedData->faceSign.juliaPtr());
        auto matrixColumnIdxV = root(cellBasedData->matrixColumnIdx.juliaPtr());
        const auto diagOffs = root(matIt->diagOffset().juliaPtr());
        const auto rowOffs = root(ls.matrix().sparsity()->rowOffs().juliaPtr());
        const auto jowners = root(mesh.faceOwners().juliaPtr());
        auto valPtr = root(ls.matrix().juliaPtr());
        jl_value_t* bValues =
            root(jl_call1((jl_function_t*)concrete_type, (jl_value_t*)ls.boundaryMatrix().juliaPtr()));
        jl_value_t* wrapped = root(jl_call1(
            (jl_function_t*)concrete_type,
            (jl_value_t*)valPtr
        ));

        size_t nbinputs = useGPU ? 29 : 27;
        jl_value_t* args[29];
        args[0] = root(jl_box_int32(iterator->size()));
        args[1] = (jl_value_t*)jowners;
        args[2] = (jl_value_t*)cellFacesSegments;
        args[3] = (jl_value_t*)diagOffs;
        args[4] = (jl_value_t*)rowOffs;
        args[5] = (jl_value_t*)cellFacesValues;
        args[6] = (jl_value_t*)faceSignV;
        // args[7] = (jl_value_t*)valPtr;
        args[7] = (jl_value_t*)wrapped;
        args[8] = root(jl_cstr_to_string(fusedOPString.c_str()));
        args[9] = (jl_value_t*)ifaceFluxV;
        args[10] = (jl_value_t*)bfaceFluxV;
        args[11] = (jl_value_t*)iGamma;
        args[12] = (jl_value_t*)bGamma;
        args[13] = (jl_value_t*)ideltaCoeffs;
        args[14] = (jl_value_t*)bdeltaCoeffs;
        args[15] = (jl_value_t*)matrixColumnIdxV;
        args[16] = (jl_value_t*)magFaceAreas;
        args[17] = (jl_value_t*)valueFraction;
        args[18] = (jl_value_t*)boundaryFaceOwners;
        args[19] = (jl_value_t*)bValues;
        args[20] = (jl_value_t*)bRhs;
        args[21] = (jl_value_t*)vol;
        args[22] = (jl_value_t*)oldVector;
        args[23] = (jl_value_t*)refValue;
        args[24] = (jl_value_t*)refGrad;
        args[25] = (jl_value_t*)rhsPtr;
        args[26] = root(jl_box_float64(dt));
        if (useGPU)
        {
            args[27] = root(jl_box_int32(nInternalFaces));
            args[28] = root(jl_box_int32(nTotalFaces));
        }
        auto funcName = useGPU ? "cellBased_gpu" : "cellBased_";
        jl_function_t* func = jl_get_function(mod, funcName);
        jl_call(func, args, static_cast<uint32_t>(nbinputs));
        detail::checkJuliaException();
        JL_GC_POP();
        return;
    }

    const auto JUowner = root(mesh.faceOwners().juliaPtr());
    const auto JUneighbour = root(mesh.faceNeighbors().juliaPtr());
    const auto JUdiagOffs = root(ls.faceToMatrixAddress()->diagOffset().juliaPtr());
    const auto JUownOffs = root(ls.faceToMatrixAddress()->ownerOffset().juliaPtr());
    const auto JUneiOffs = root(ls.faceToMatrixAddress()->neighbourOffset().juliaPtr());
    const auto JUrowOffs = root(ls.matrix().sparsity()->rowOffs().juliaPtr());
    auto valPtr = root(ls.matrix().values().juliaPtr());
    jl_value_t* bValues = root(jl_call1(
        (jl_function_t*)concrete_type, (jl_value_t*)ls.boundaryMatrix().values().juliaPtr()
    ));
    jl_value_t* wrapped = root(jl_call1(
        (jl_function_t*)concrete_type,
        (jl_value_t*)valPtr
    ));
    // the batched strategy appends the batch segments of the BatchedFaceIterator
    const bool batched = strategy == Strategy::Batched || strategy == Strategy::BatchedCUDA;
    NF_ASSERT(strategy != Strategy::BatchedCUDA || useGPU, "BatchedCUDA needs a GPU executor");
    std::shared_ptr<la::BatchedFaceIterator::BatchedFaceData> batchedData;
    if (batched)
    {
        NF_ASSERT(batchedIterator, "JuNe batched assembly needs a BatchedFaceIterator");
        if (!batchedIterator->getBatchedFaceData())
        {
            batchedIterator->setComputeBatchedFaceData(mesh);
        }
        batchedData = batchedIterator->getBatchedFaceData();
    }

    size_t nbinputs = 26 + (batched ? 3 : 0) + (useGPU ? 2 : 0);
    jl_value_t* args[31];
    args[0] = root(jl_box_int32(nInternalFaces));
    args[1] = (jl_value_t*)JUowner;
    args[2] = (jl_value_t*)JUneighbour;
    args[3] = (jl_value_t*)JUdiagOffs;
    args[4] = (jl_value_t*)JUownOffs;
    args[5] = (jl_value_t*)JUneiOffs;
    args[6] = (jl_value_t*)JUrowOffs;
    // args[7] = (jl_value_t*)valPtr;
    args[7] = (jl_value_t*)wrapped;
    args[8] = root(jl_cstr_to_string(fusedOPString.c_str()));
    args[9] = (jl_value_t*)ifaceFluxV;
    args[10] = (jl_value_t*)bfaceFluxV;
    args[11] = (jl_value_t*)iGamma;
    args[12] = (jl_value_t*)bGamma;
    args[13] = (jl_value_t*)ideltaCoeffs;
    args[14] = (jl_value_t*)bdeltaCoeffs;
    args[15] = (jl_value_t*)magFaceAreas;
    args[16] = (jl_value_t*)valueFraction;
    args[17] = (jl_value_t*)refValue;
    args[18] = (jl_value_t*)refGrad;
    args[19] = (jl_value_t*)rhsPtr;
    args[20] = (jl_value_t*)boundaryFaceOwners;
    args[21] = (jl_value_t*)bValues;
    args[22] = (jl_value_t*)bRhs;
    args[23] = (jl_value_t*)vol;
    args[24] = (jl_value_t*)oldVector;
    args[25] = root(jl_box_float64(dt));
    size_t next = 26;
    if (batched)
    {
        auto [batchFaces, batchSegments] = batchedData->batchFaces.juliaPtrs();
        root(batchFaces);
        root(batchSegments);
        args[next++] = root(jl_box_int32(batchedData->nBatches));
        args[next++] = batchSegments;
        args[next++] = batchFaces;
    }
    if (useGPU)
    {
        args[next++] = root(jl_box_int32(nCells));
        args[next++] = root(jl_box_int32(nTotalFaces));
    }
    const bool global = strategy == Strategy::GlobalFace;
    const char* funcName = strategy == Strategy::BatchedCUDA ? "assemble_batched_cuda"
                         : batched ? (useGPU ? "assemble_batched_gpu" : "assemble_batched")
                         : useGPU  ? (global ? "globalassemble_gpu" : "assemble_gpu")
                                   : (global ? "globalassemble" : "assemble");
    jl_function_t* func = jl_get_function(mod, funcName);
    jl_call(func, args, static_cast<uint32_t>(nbinputs));
    detail::checkJuliaException();
    }
    catch (...)
    {
        JL_GC_POP();
        throw;
    }
    JL_GC_POP();
}

/* @brief compile the JuNe face-based kernels for the operators of expr ahead of timing */
template<typename ValueType, typename IndexType>
void warmup(const Expression<ValueType, IndexType>& expr)
{
    namespace fvcc = NeoN::finiteVolume::cellCentred;
    // the GPU kernels are compiled on a two-cell mesh when the equation lives on the GPU
    const auto* divPhi =
        detail::findConfig<fvcc::VolumeField<ValueType>>(expr, "DivOperator", "field");
    const auto* lapPhi =
        detail::findConfig<fvcc::VolumeField<ValueType>>(expr, "LaplacianOperator", "field");
    NF_ASSERT(divPhi || lapPhi, "JuNe assembly needs a div or laplacian operator");
    const bool useGPU = std::holds_alternative<GPUExecutor>((divPhi ? *divPhi : *lapPhi).exec());
    jl_module_t* mod = (jl_module_t*)jl_eval_string("JuNe");
    jl_function_t* func = jl_get_function(mod, useGPU ? "warmup_gpu" : "warmup");
    jl_call1(func, jl_cstr_to_string(opString(expr).c_str()));
    detail::checkJuliaException();
}

} // namespace june

} // namespace NeoN::dsl

#endif // NeoN_WITH_JULIA
