// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoN authors
// inspired by
// https://develop.openfoam.com/modules/external-solver/-/blob/develop/src/petsc4Foam/utils/petscLinearSolverContext.H

#pragma once

#if NF_WITH_PETSC

#include <Kokkos_Core.hpp>
#include <petscvec_kokkos.hpp>
#include <petscmat.h>
#include <petscksp.h>

#include "NeoN/fields/field.hpp"
#include "NeoN/core/dictionary.hpp"
#include "NeoN/linearAlgebra/linearSystem.hpp"
#include "NeoN/linearAlgebra/utilities.hpp"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace NeoN::la::petscSolverContext
{

template<typename ValueType>
class petscSolverContext
{
    // Private Data

    bool init_, updated_;
    Executor exec_;
    Mat Amat_;
    KSP ksp_;
    PC pc_;

    Vec sol_, rhs_;

    Dictionary solverDict_;

public:


    // Constructors

    //- Default construct
    petscSolverContext(Executor exec, Dictionary solverDict)
        : init_(false), updated_(false), exec_(exec), Amat_(nullptr), ksp_(nullptr), pc_(nullptr),
          sol_(nullptr), rhs_(nullptr), solverDict_(solverDict)
    {}


    //- Destructor
    virtual ~petscSolverContext()
    {
        MatDestroy(&Amat_);
        KSPDestroy(&ksp_);
        VecDestroy(&sol_);
        VecDestroy(&rhs_);
    }


    // Member Functions

    //- Return value of initialized
    bool initialized() const noexcept { return init_; }


    //- Return value of updated
    bool updated() const noexcept { return updated_; }

    //- Create auxiliary rows for calculation purposes
    void initialize(const LinearSystem<scalar, localIdx>& sys)
    {

        // move all not necessary staff to outer most scope since matrix  has
        // to be preallocated only once every time the mesh changes
        PetscInitialize(NULL, NULL, 0, NULL);

        setOption(solverDict_);

        std::cout << sys.matrix().rowOffs().size() << std::endl;
        std::cout << sys.matrix().colIdxs().size() << std::endl;
        auto rowPtrHost = sys.matrix().rowOffs().copyToHost();
        auto rowPtrHostv = rowPtrHost.view();

        auto colIdxHost = sys.matrix().colIdxs().copyToHost();
        auto colIdxHostv = colIdxHost.view();

        localIdx sizeMatrix = static_cast<localIdx>(sys.matrix().values().size());
        localIdx nrows = sys.rhs().size();

        PetscInt *colIdx, *rowIdx, *rhsIdx;

        PetscMalloc1(static_cast<PetscInt>(sizeMatrix), &colIdx);
        PetscMalloc1(static_cast<PetscInt>(sizeMatrix), &rowIdx);
        PetscMalloc1(static_cast<PetscInt>(nrows), &rhsIdx);

        // PetscInt colIdx[sizeMatrix];
        // PetscInt rowIdx[sizeMatrix];
        // PetscInt rhsIdx[nrows];


        for (int index = 0; index < nrows; ++index)
        {
            rhsIdx[index] = static_cast<PetscInt>(index);
        }
        // copy colidx
        // TODO: (this should be done only once when the matrix
        //  topology changes
        for (int index = 0; index < sizeMatrix; ++index)
        {
            colIdx[index] = static_cast<PetscInt>(colIdxHostv[index]);
        }
        // convert rowPtr to rowIdx
        // TODO: (this should be done only once when the matrix
        //  topology changes
        localIdx rowI = 0;
        localIdx rowOffset = rowPtrHostv[rowI + 1];
        for (int index = 0; index < sizeMatrix; ++index)
        {
            if (index == rowOffset)
            {
                rowI++;
                rowOffset = rowPtrHostv[rowI + 1];
            }
            rowIdx[index] = rowI;
        }


        MatCreate(PETSC_COMM_WORLD, &Amat_);
        MatSetSizes(Amat_, sys.matrix().nRows(), sys.rhs().size(), PETSC_DECIDE, PETSC_DECIDE);

        VecCreate(PETSC_COMM_SELF, &rhs_);
        VecSetSizes(rhs_, PETSC_DECIDE, nrows);

        std::string execName = std::visit([](const auto& e) { return e.name(); }, exec_);

        if (execName == "GPUExecutor")
        {
            VecSetType(rhs_, VECKOKKOS);
            MatSetType(Amat_, MATAIJKOKKOS);
        }
        else
        {
            VecSetType(rhs_, VECSEQ);
            MatSetType(Amat_, MATSEQAIJ);
        }
        VecDuplicate(rhs_, &sol_);

        VecSetPreallocationCOO(rhs_, nrows, rhsIdx);
        MatSetPreallocationCOO(Amat_, sizeMatrix, colIdx, rowIdx);

        KSPCreate(PETSC_COMM_WORLD, &ksp_);
        KSPSetFromOptions(ksp_);
        KSPSetOperators(ksp_, Amat_, Amat_);


        init_ = true;

        // PetscOptions options;
        // PetscOptionsCreate(&options);
        // PetscOptionsSetValue(NULL, "-no_signal_handler", "true");
        PetscOptionsView(NULL, PETSC_VIEWER_STDOUT_WORLD);
        KSPView(ksp_, PETSC_VIEWER_STDOUT_WORLD);


        PetscFree(colIdx);
        PetscFree(rowIdx);
        PetscFree(rhsIdx);
    }

    //- Create auxiliary rows for calculation purposes
    void update() { NF_ERROR_EXIT("Mesh changes not supported"); }

    void setOption(Dictionary& solverDict)
    {

        NeoN::Dictionary subDict = solverDict.subDict("options");

        for (auto key : solverDict.subDict("options").keys())
        {

            std::string petscOptionKey = std::string("-") + key;
            std::string petscOptionVal = subDict.get<std::string>(key);
            PetscOptionsSetValue(NULL, petscOptionKey.c_str(), petscOptionVal.c_str());
        }
    }

    std::string getOption(std::string optionName)
    {
        char optionValue[PETSC_MAX_PATH_LEN];
        PetscBool set;
        PetscOptionsGetString(
            NULL, NULL, optionName.c_str(), optionValue, sizeof(optionValue), &set
        );

        std::string optionValueStr(optionValue);

        // TODO: Decide what to do (error or warning) if set is FALSE

        return optionValueStr;
    }

    [[nodiscard]] Mat& AMat() { return Amat_; }

    [[nodiscard]] Vec& rhs() { return rhs_; }

    [[nodiscard]] Vec& sol() { return sol_; }

    [[nodiscard]] KSP& ksp() { return ksp_; }

    [[nodiscard]] PC& pc() { return pc_; }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End NeoN::la::petscSolverContext

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif
