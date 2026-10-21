// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <variant>

#include <julia.h>

#include "NeoN/core/dictionary.hpp"
#include "NeoN/core/error.hpp"
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
    FromIterator, // CellBasedIterator on the linear system -> Cell, anything else -> Face
    Face,
    GlobalFace,
    Cell
};

/* @brief whether JuNe implements every implicit operator of the expression */
template<typename ValueType, typename IndexType>
bool canAssemble(const Expression<ValueType, IndexType>& expr)
{
    bool hasSpatial = false;
    for (const auto& op : expr.spatialOperators())
    {
        if (op.getType() != Operator::Type::Implicit) continue;
        const auto name = op.getName();
        const auto jlOp = op.juliaOP();
        // JuNe only defines the upwind, linearUpwind and linear div schemes
        const bool supportedDiv = name == "DivOperator"
                               && (jlOp.starts_with("Div{Float64, upwind{")
                                   || jlOp.starts_with("Div{Float64, linearUpwind{")
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
        const char* p = jl_string_ptr(
            jl_eval_string("sprint(showerror, ccall(:jl_exception_occurred, Any, ()))")
        );
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
template<typename ValueType, typename IndexType>
void assemble(
    const Expression<ValueType, IndexType>& expr,
    la::LinearSystem<ValueType, ValueType>& ls,
    scalar dt,
    Strategy strategy = Strategy::FromIterator
)
{
    namespace fvcc = NeoN::finiteVolume::cellCentred;

    const auto* divPhi = detail::findConfig<fvcc::VolumeField<ValueType>>(expr, "DivOperator", "field");
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

    const auto ifaceFluxV = faceFlux ? faceFlux->internalVector().juliaPtr() : empty.juliaPtr();
    const auto bfaceFluxV =
        faceFlux ? faceFlux->boundaryData().value().juliaPtr() : empty.juliaPtr();
    const auto iGamma = gamma ? gamma->internalVector().juliaPtr() : empty.juliaPtr();
    const auto bGamma = gamma ? gamma->boundaryData().value().juliaPtr() : empty.juliaPtr();
    const auto oldVector =
        hasDdt ? fvcc::oldTime(phi).internalVector().juliaPtr() : emptyOld.juliaPtr();

    const auto nInternalFaces = mesh.nInternalFaces();
    const auto nTotalFaces = mesh.nTotalFaces();
    const auto nCells = mesh.nCells();
    bool useGPU = std::holds_alternative<GPUExecutor>(mesh.exec());

    // the cached geometry-scheme coefficients: the mesh centres they are computed from are
    // released after the first geometry read
    const auto& nonOrthDeltaCoeffs =
        fvcc::GeometryScheme::readOrCreate(mesh)->nonOrthDeltaCoeffs();
    const auto ideltaCoeffs = nonOrthDeltaCoeffs.internalVector().juliaPtr();
    const auto bdeltaCoeffs = nonOrthDeltaCoeffs.boundaryData().value().juliaPtr();

    const auto magFaceAreas = mesh.faceAreas().juliaPtr();
    const auto boundaryFaceOwners = mesh.boundaryMesh().faceOwners().juliaPtr();
    const auto vol = mesh.cellVolumes().juliaPtr();
    auto refGrad = phi.boundaryData().refGrad().juliaPtr();
    auto valueFraction = phi.boundaryData().valueFraction().juliaPtr();
    auto refValue = phi.boundaryData().refValue().juliaPtr();
    auto rhsPtr = ls.rhs().juliaPtr();
    auto bRhs = ls.boundaryRhs().juliaPtr();

    auto iterator = std::dynamic_pointer_cast<la::CellBasedIterator>(ls.getMeshIterator()->get());
    if (strategy == Strategy::FromIterator)
    {
        strategy = iterator ? Strategy::Cell : Strategy::Face;
    }

    jl_module_t* mod = (jl_module_t*)jl_eval_string("JuNe");

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
        auto faceSignV = cellBasedData->faceSign.juliaPtr();
        auto matrixColumnIdxV = cellBasedData->matrixColumnIdx.juliaPtr();
        const auto diagOffs = matIt->diagOffset().juliaPtr();
        const auto rowOffs = ls.matrix().sparsity()->rowOffs().juliaPtr();
        const auto jowners = mesh.faceOwners().juliaPtr();
        auto valPtr = ls.matrix().juliaPtr();
        auto bValues = ls.boundaryMatrix().juliaPtr();

        size_t nbinputs = useGPU ? 29 : 27;
        jl_value_t* args[29];
        args[0] = jl_box_int32(iterator->size());
        args[1] = (jl_value_t*)jowners;
        args[2] = (jl_value_t*)cellFacesSegments;
        args[3] = (jl_value_t*)diagOffs;
        args[4] = (jl_value_t*)rowOffs;
        args[5] = (jl_value_t*)cellFacesValues;
        args[6] = (jl_value_t*)faceSignV;
        args[7] = (jl_value_t*)valPtr;
        args[8] = jl_cstr_to_string(fusedOPString.c_str());
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
        args[26] = jl_box_float64(dt);
        if (useGPU)
        {
            args[27] = jl_box_int32(nInternalFaces);
            args[28] = jl_box_int32(nTotalFaces);
        }
        auto funcName = useGPU ? "cellBased_gpu" : "cellBased_";
        jl_function_t* func = jl_get_function(mod, funcName);
        jl_call(func, args, static_cast<uint32_t>(nbinputs));
        detail::checkJuliaException();
        return;
    }

    const auto JUowner = mesh.faceOwners().juliaPtr();
    const auto JUneighbour = mesh.faceNeighbors().juliaPtr();
    const auto JUdiagOffs = ls.faceToMatrixAddress()->diagOffset().juliaPtr();
    const auto JUownOffs = ls.faceToMatrixAddress()->ownerOffset().juliaPtr();
    const auto JUneiOffs = ls.faceToMatrixAddress()->neighbourOffset().juliaPtr();
    const auto JUrowOffs = ls.matrix().sparsity()->rowOffs().juliaPtr();
    auto valPtr = ls.matrix().values().juliaPtr();
    auto bValues = ls.boundaryMatrix().values().juliaPtr();

    size_t nbinputs = useGPU ? 28 : 26;
    jl_value_t* args[28];
    args[0] = jl_box_int32(nInternalFaces);
    args[1] = (jl_value_t*)JUowner;
    args[2] = (jl_value_t*)JUneighbour;
    args[3] = (jl_value_t*)JUdiagOffs;
    args[4] = (jl_value_t*)JUownOffs;
    args[5] = (jl_value_t*)JUneiOffs;
    args[6] = (jl_value_t*)JUrowOffs;
    args[7] = (jl_value_t*)valPtr;
    args[8] = jl_cstr_to_string(fusedOPString.c_str());
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
    args[25] = jl_box_float64(dt);
    if (useGPU)
    {
        args[26] = jl_box_int32(nCells);
        args[27] = jl_box_int32(nTotalFaces);
    }
    const bool global = strategy == Strategy::GlobalFace;
    auto funcName = useGPU ? (global ? "globalassemble_gpu" : "assemble_gpu")
                           : (global ? "globalassemble" : "assemble");
    jl_function_t* func = jl_get_function(mod, funcName);
    jl_call(func, args, static_cast<uint32_t>(nbinputs));
    detail::checkJuliaException();
}

/* @brief compile the JuNe face-based kernels for the operators of expr ahead of timing */
template<typename ValueType, typename IndexType>
void warmup(const Expression<ValueType, IndexType>& expr)
{
    jl_module_t* mod = (jl_module_t*)jl_eval_string("JuNe");
    jl_function_t* func = jl_get_function(mod, "warmup");
    jl_call1(func, jl_cstr_to_string(opString(expr).c_str()));
    detail::checkJuliaException();
}

} // namespace june

} // namespace NeoN::dsl
