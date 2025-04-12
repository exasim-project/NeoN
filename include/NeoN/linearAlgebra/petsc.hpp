// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 NeoFOAM authors

#pragma once

#if NF_WITH_PETSC

#include <Kokkos_Core.hpp>
#include <petscvec_kokkos.hpp>
#include <petscmat.h>
#include <petscksp.h>

#include "NeoFOAM/fields/field.hpp"
#include "NeoFOAM/core/dictionary.hpp"
#include "NeoFOAM/linearAlgebra/linearSystem.hpp"
#include "NeoFOAM/linearAlgebra/utilities.hpp"
#include "NeoFOAM/linearAlgebra/petscSolverContext.hpp"
#include "NeoFOAM/linearAlgebra/solver.hpp"
// #include "NeoFOAM/core/database/petscSolverContextCollection.hpp"

namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

namespace NeoFOAM::la::petsc
{

// template<typename ValueType>

class petscSolver : public SolverFactory::template Register<petscSolver>
{

    using Base = SolverFactory::template Register<petscSolver>;

private:

    Dictionary solverDict_;
    // Mat Amat_;
    // KSP ksp_;

    // Vec sol_, rhs_;

    // NeoFOAM::la::petscSolverContext::petscSolverContext<scalar> petsctx_;

    /*
        NeoFOAM::Database& db_;
        std::string eqnName_;
    */

public:

    petscSolver(Executor exec, Dictionary solverDict) : Base(exec), solverDict_(solverDict)
    //, Amat_(nullptr), ksp_(nullptr), sol_(nullptr),
    //  rhs_(nullptr),petsctx_(exec_)
    //, db_(db), eqnName_(eqnName)
    {}

    //- Destructor
    virtual ~petscSolver()
    {
        // MatDestroy(&Amat_);
        // KSPDestroy(&ksp_);
        // VecDestroy(&sol_);
        // VecDestroy(&rhs_);
    }

    static std::string name() { return "Petsc"; }

    static std::string doc() { return "TBD"; }

    static std::string schema() { return "none"; }

    virtual std::unique_ptr<SolverFactory> clone() const final
    {
        // FIXME
        return {};
    }


    virtual void solve(const LinearSystem<scalar, localIdx>& sys, Field<scalar>& x) const final
    {

        Mat Amat;
        KSP ksp;

        Vec sol, rhs;

        NeoFOAM::la::petscSolverContext::petscSolverContext<scalar> petsctx(exec_);

        std::size_t nrows = sys.rhs().size();

        petsctx.initialize(sys);

        Amat = petsctx.AMat();
        rhs = petsctx.rhs();
        sol = petsctx.sol();
        ksp = petsctx.ksp();

        VecSetValuesCOO(rhs, sys.rhs().data(), ADD_VALUES);
        MatSetValuesCOO(Amat, sys.matrix().values().data(), ADD_VALUES);

        // MatView(Amat_, PETSC_VIEWER_STDOUT_WORLD);
        // VecView(rhs_, PETSC_VIEWER_STDOUT_WORLD);


        // KSPCreate(PETSC_COMM_WORLD, &ksp_);
        KSPSetOperators(ksp, Amat, Amat);
        // KSPSetTolerances(ksp, 1.e-9, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
        // KSPSetFromOptions(ksp_);
        // KSPSetUp(ksp);


        PetscCallVoid(KSPSolve(ksp, rhs, sol));


        PetscScalar* x_help = static_cast<PetscScalar*>(x.data());
        VecGetArray(sol, &x_help);

        NeoFOAM::Field<NeoFOAM::scalar> x2(x.exec(), static_cast<scalar*>(x_help), nrows, x.exec());
        x = x2;
    }
};

}

#endif
