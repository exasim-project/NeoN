// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"
#include <julia.h>
JULIA_DEFINE_FAST_TLS; // only define this once, in an executable (not in a
// shared library) if you want fast code.
#include "NeoN/NeoN.hpp"
#include "NeoN/core/primitives/vec3.hpp"
using namespace NeoN;
#include "NeoN/core/vector/vectorFreeFunctions.hpp"


TEST_CASE("GPUExec single vector")
{
    jl_init();
    auto exec = GPUExecutor {};
	jl_eval_string("include(\"../../../../init.jl\")");  // NeoN root directory relative to where this compiles to
    jl_eval_string(R"(
      	function use_pointer(ptr::Ptr{Cvoid}, size::Int64)
      		vec = unsafe_wrap(CuArray, reinterpret(CuPtr{Float64}, ptr),size)
      		vec[1:3] = [1.0,2.0,3.0]    
      	end
    )");
	int64_t  N = 10;
    auto input = Vector<double>(exec, N, 0.0);
   
    
    auto ptrval = jl_box_voidpointer((void*)input.data());
    auto boxed_size = jl_box_int64(N);
	jl_value_t* args[2];
	args[0] = (jl_value_t*)ptrval;
	args[1] = (jl_value_t*)boxed_size;
    jl_function_t* func = jl_get_function(jl_main_module, "use_pointer");
    jl_call(func, args, 2);
    if (jl_exception_occurred())
    {
       const char* p = jl_string_ptr(
           jl_eval_string("sprint(showerror, ccall(:jl_exception_occurred, Any, ()))")
       );

       fprintf(stderr, "%s%s\n", "error: ", p);
    }    
	REQUIRE(!jl_exception_occurred());
    auto cpu = CPUExecutor {};
	auto cpuarr = input.copyToExecutor(cpu);
	auto view = cpuarr.view();
	REQUIRE(view[0] == 1.0);
	REQUIRE(view[1] == 2.0);
	REQUIRE(view[2] == 3.0);
	jl_atexit_hook(0);
}

TEST_CASE("GPUExec vec<vec3> 1")
{
    jl_init();
    auto exec = GPUExecutor {};
    jl_eval_string("include(\"../../../../init.jl\")");  // NeoN root directory relative to where this compiles to
    jl_eval_string(R"(
        function use_pointer(addr::UInt64, size::Int64)
            ptr = CuPtr{Float64}(addr)
			vec = unsafe_wrap(CuArray, ptr, (size,3); own=false)
            #vec[1:3] = [21.0,22.0,23.0]
			#vec[1:3] = [1.0,2.0,3.0] # start
			#vec[22:24] = [22.0, 23.0, 24.0]
        end
    )");
    int64_t  N = 10;
    auto input = Vector<Vec3>(exec, N, zero<Vec3>());


    auto ptrval = jl_box_voidpointer((void*)input.data());
    auto boxed_size = jl_box_int64(N);
    jl_value_t* args[2];
    args[0] = (jl_value_t*)ptrval;
    args[1] = (jl_value_t*)boxed_size;
    
	JL_GC_PUSH2(args[0], args[1]);
	jl_function_t* func = jl_get_function(jl_main_module, "use_pointer");
    jl_call(func, args, 2);
    if (jl_exception_occurred())
    {
       const char* p = jl_string_ptr(
           jl_eval_string("sprint(showerror, ccall(:jl_exception_occurred, Any, ()))")
       );

       fprintf(stderr, "%s%s\n", "error: ", p);
    }
    REQUIRE(!jl_exception_occurred());
    auto cpu = CPUExecutor {};
    auto cpuarr = input.copyToExecutor(cpu);
    auto view = cpuarr.view();
    REQUIRE(view[0][0] == 21.0);
	REQUIRE(view[0][1] == 22.0);
    REQUIRE(view[0][2] == 23.0);
	JL_GC_POP();
	jl_atexit_hook(0);
}

TEST_CASE("Julia Hello World")
{
    jl_init();
    jl_eval_string("println(\"Hello from NEON!\")");
    jl_atexit_hook(0);
}
TEST_CASE("[CPU] Pass Vector<Vec3> to Julia")
{
    jl_init();
    auto exec = Executor(CPUExecutor {});
    jl_eval_string(R"(
        function use_pointer(x::Vector{Float64})
            x[1:3] = [1.0, 2.0, 3.0]
        end
    )");
    auto input = Vector<Vec3>(exec, 10, zero<Vec3>());
    auto julia2dptr = input.juliaPtr();
    jl_function_t* func = jl_get_function(jl_main_module, "use_pointer");
    jl_call1(func, (jl_value_t*)julia2dptr);
    if (jl_exception_occurred())
    {
        std::cerr << "Julia exception: " << jl_typeof_str(jl_exception_occurred()) << std::endl;
    }
    auto view = input.view();
	REQUIRE(view[0][0] == 1.0);
    REQUIRE(view[0][1] == 2.0);
    REQUIRE(view[0][2] == 3.0);

    jl_atexit_hook(0);
}

TEST_CASE("[CPU] Pass NeoN::Array to Julia and Receive")
{
    jl_init();
    auto exec = NeoN::Executor(NeoN::CPUExecutor {});
    jl_eval_string(R"(
        function use_pointer(p::Ptr{Cvoid})
            arr = unsafe_wrap(Array, Ptr{Float64}(p), 10)
            arr[1] = 42.0
            return nothing
        end
    )");

    size_t size = 10;
    NeoN::Array<NeoN::scalar> array(exec, size);
    jl_value_t* array_type = jl_apply_array_type((jl_value_t*)jl_float64_type, 1);
    jl_value_t* julia_ptr = jl_box_voidpointer((void*)array.data());
    jl_function_t* func = jl_get_function(jl_main_module, "use_pointer");

    jl_call1(func, julia_ptr);
    REQUIRE(array.view()[0] == 42.0);

    jl_atexit_hook(0);
}

// TODO GPU

TEST_CASE("Pass multiple NeoN::Array to Julia and Receive")
{
    jl_init();
    auto exec = NeoN::Executor(NeoN::SerialExecutor {});
    jl_eval_string(R"(
        function use_pointer(a::Ptr{Cvoid}, b::Ptr{Cvoid})
            arr = unsafe_wrap(Array, Ptr{Float64}(a), 10)
            arr2 = unsafe_wrap(Array, Ptr{Float64}(b), 10)
            arr[1] = 42.0
            arr2[1] = 69.0
            return nothing
        end
    )");

    size_t size = 10;
    NeoN::Array<NeoN::scalar> array(exec, size);
    NeoN::Array<NeoN::scalar> array2(exec, size);
    jl_value_t* array_type = jl_apply_array_type((jl_value_t*)jl_float64_type, 1);
    jl_value_t* julia_ptr = jl_box_voidpointer((void*)array.data());
    jl_value_t* julia_ptr2 = jl_box_voidpointer((void*)array2.data());
    jl_function_t* func = jl_get_function(jl_main_module, "use_pointer");
    jl_value_t* args[2];
    args[0] = julia_ptr;
    args[1] = julia_ptr2;
    jl_call(func, args, 2);
    REQUIRE(array.view()[0] == 42.0);
    REQUIRE(array2.view()[0] == 69.0);

    jl_atexit_hook(0);
}


TEST_CASE("Pass string to julia")
{
    jl_init();
    jl_eval_string(R"(
        function pass_string(p::String)
            return uppercase(p)
        end
    )");

    jl_function_t* func = jl_get_function(jl_main_module, "pass_string");
    std::string julia = "julia";
    jl_value_t* argument = jl_cstr_to_string(julia.c_str());
    jl_value_t* ret = jl_call1(func, argument);
    const char* unboxed = jl_string_ptr(ret);
    std::string str = std::string(unboxed);
    REQUIRE("JULIA" == str);
    jl_atexit_hook(0);
}

TEST_CASE("Pass std::vector<std::string> to julia and receive scheme string")
{
    jl_init();
    jl_eval_string(R"(
        function pass_strings(p::Vector{String})
            return p[2]
        end
    )");
    NeoN::TokenList input = NeoN::TokenList({std::string("Gauss"), std::string("linear")});

    jl_value_t* array_type = jl_apply_array_type((jl_value_t*)jl_string_type, 1);

    jl_array_t* jl_vec = jl_alloc_array_1d(array_type, input.size());

    JL_GC_PUSH1(&jl_vec); // basically: increment GC reference so it doesnt get free'd

    for (size_t i = 0; i < input.tokens().size(); ++i)
    {
        jl_value_t* s = jl_cstr_to_string(input.get<std::string>(i).c_str());
        jl_array_ptr_set(jl_vec, i, s);
    }

    jl_function_t* func = jl_get_function(jl_main_module, "pass_strings");
    jl_value_t* ret = jl_call1(func, (jl_value_t*)jl_vec);
    const char* p = jl_string_ptr(ret);
    std::string returnOp(p);
    REQUIRE(input.get<std::string>(1) == returnOp);
    JL_GC_POP(); // decrement reference
    jl_atexit_hook(0);
}


TEST_CASE("Pass Div tokens and create Julia Div Operator")
{
    jl_init();
    jl_eval_string(R"(
        abstract type DivScheme{P} end

        struct UpwindScheme{P} <: DivScheme{P} end
        @inline (s::UpwindScheme{P})(ϕ) where {P<:AbstractFloat} = ifelse(ϕ ≥ 0, one(P), zero(P))

        struct CentralDiffScheme{P} <: DivScheme{P} end
        @inline (u::CentralDiffScheme{P})(ϕf) where {P<:AbstractFloat} = P(0.5)

        struct Div{P,S}
            scheme::S
            scale::P
        end

        function pass_div(tokens::Vector{String})
            divscheme = ifelse(tokens[2] == "linear", CentralDiffScheme{Float64},UpwindScheme{Float64}) 
            div = Div{Float64,divscheme}(divscheme(), 1.0) 
            return String(Symbol(div))
        end
    )");
    NeoN::TokenList input = NeoN::TokenList({std::string("Gauss"), std::string("linear")});

    jl_value_t* array_type = jl_apply_array_type((jl_value_t*)jl_string_type, 1);

    jl_array_t* jl_vec = jl_alloc_array_1d(array_type, input.size());

    JL_GC_PUSH1(&jl_vec); // basically: increment GC reference so it doesnt get free'd

    for (size_t i = 0; i < input.tokens().size(); ++i)
    {
        jl_value_t* s = jl_cstr_to_string(input.get<std::string>(i).c_str());
        jl_array_ptr_set(jl_vec, i, s);
    }

    jl_function_t* func = jl_get_function(jl_main_module, "pass_div");
    jl_value_t* ret = jl_call1(func, (jl_value_t*)jl_vec);
    if (jl_exception_occurred())
    {
        std::cerr << "Julia exception: " << jl_typeof_str(jl_exception_occurred()) << std::endl;
    }
    const char* p = jl_string_ptr(ret);
    std::string returnOp(p);
    REQUIRE(
        returnOp == "Div{Float64, CentralDiffScheme{Float64}}(CentralDiffScheme{Float64}(), 1.0)"
    );
    JL_GC_POP(); // decrement reference
    jl_atexit_hook(0);
}


namespace fvcc = NeoN::finiteVolume::cellCentred;

using Operator = NeoN::dsl::Operator;

namespace NeoN
{


TEMPLATE_TEST_CASE("DivOperator", "[template]", NeoN::scalar, NeoN::Vec3)
{
    auto [execName, exec] = GENERATE(allAvailableExecutor());

    auto mesh = create1DUniformMesh(exec, 10);
    auto surfaceBCs = fvcc::createCalculatedBCs<fvcc::SurfaceBoundary<scalar>>(mesh);

    // compute corresponding uniform faceFlux
    // TODO this should be handled outside of the unit test
    fvcc::SurfaceField<scalar> faceFlux(exec, "sf", mesh, surfaceBCs);
    fill(faceFlux.internalVector(), 1.0);
    auto boundFaceFlux = faceFlux.internalVector().view();
    // face on the left side has different orientation
    parallelFor(
        exec, {mesh.nCells() - 1, mesh.nCells()}, NEON_LAMBDA(const localIdx i) {
            boundFaceFlux[i] = -1.0;
        }
    );

    auto volumeBCs = fvcc::createCalculatedBCs<fvcc::VolumeBoundary<TestType>>(mesh);
    fvcc::VolumeField<TestType> phi(exec, "sf", mesh, volumeBCs);
    fill(phi.internalVector(), one<TestType>());
    fill(phi.boundaryData().value(), one<TestType>());
    phi.correctBoundaryConditions();

    auto result = Vector<TestType>(exec, phi.size());
    fill(result, zero<TestType>());

    SECTION("Construct from Token" + execName)
    {
        Input input = TokenList({std::string("Gauss"), std::string("linear")});
        fvcc::DivOperator(Operator::Type::Explicit, faceFlux, phi, input);
    }

    SECTION("Construct from Dictionary" + execName)
    {
        Input input = Dictionary(
            {{std::string("DivOperator"), std::string("Gauss")},
             {std::string("surfaceInterpolation"), std::string("linear")}}
        );
        auto op = fvcc::DivOperator(Operator::Type::Explicit, faceFlux, phi, input);
        op.divJulia(result);

        // divergence of a uniform field should be zero
        auto outHost = result.copyToHost();
        auto outHostView = outHost.view();
        for (int i = 0; i < result.size(); i++)
        {
            REQUIRE(outHostView[i] == zero<TestType>());
        }
    }
}

}
