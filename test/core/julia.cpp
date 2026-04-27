// SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include "catch2_common.hpp"
#include <julia.h>
// JULIA_DEFINE_FAST_TLS // only define this once, in an executable (not in a
// shared library) if you want fast code.
#include "NeoN/NeoN.hpp"

TEST_CASE("Julia Hello World")
{
    jl_init();
    jl_eval_string("println(\"Hello from NEON!\")");
    jl_atexit_hook(0);
}

TEST_CASE("Include Julia File")
{
    jl_init();
    jl_eval_string("include(\"test/core/julia/test_arrays.jl\")");
    jl_atexit_hook(0);
}

TEST_CASE("[CPU] Pass NeoN::Array to Julia and Receive")
{
    jl_init();
    auto exec = NeoN::Executor(NeoN::CPUExecutor {});
    jl_eval_string("include(\"../../../../test/core/julia/test_arrays.jl\")");

    size_t size = 10;
    NeoN::Array<NeoN::scalar> array(exec, size);
    jl_value_t* array_type = jl_apply_array_type((jl_value_t*)jl_float64_type, 1);
    jl_value_t* julia_ptr = jl_box_voidpointer((void*)array.data());
    jl_function_t* func = jl_get_function(jl_main_module, "use_pointer");

    jl_call1(func, julia_ptr);
    REQUIRE(array.view()[0] == 42.0);

    jl_atexit_hook(0);
}

TEST_CASE("[SERIAL] Pass NeoN::Array to Julia and Receive")
{
    jl_init();
    auto exec = NeoN::Executor(NeoN::SerialExecutor {});
    jl_eval_string("include(\"../../../../test/core/julia/test_arrays.jl\")");

    size_t size = 10;
    NeoN::Array<NeoN::scalar> array(exec, size);
    jl_value_t* array_type = jl_apply_array_type((jl_value_t*)jl_float64_type, 1);
    jl_value_t* julia_ptr = jl_box_voidpointer((void*)array.data());
    jl_function_t* func = jl_get_function(jl_main_module, "use_pointer");

    jl_call1(func, julia_ptr);
    REQUIRE(array.view()[0] == 42.0);

    jl_atexit_hook(0);
}

// TEST_CASE("[GPU] Pass NeoN::Array to Julia and Receive")
// {
//     jl_init();
//     auto exec = NeoN::Executor(NeoN::GPUExecutor {});
//     jl_eval_string("include(\"/home/peter/Documents/uni/FVM-Prototyping/init.jl\")");

//     jl_eval_string(
//         "include(\"/home/peter/clones/neoninstall/NeoN/test/core/julia/test_arrays.jl\")"
//     );
//     jl_module_t* mod = jl_main_module;

//     size_t size = 10;
//     NeoN::Array<NeoN::scalar> array(exec, size);
//     jl_value_t* array_type = jl_apply_array_type((jl_value_t*)jl_float64_type, 1);
//     jl_value_t* julia_ptr = jl_box_voidpointer((void*)array.data());
//     jl_function_t* func = jl_get_function(jl_main_module, "use_pointer_gpu");

//     jl_call1(func, julia_ptr);
//     REQUIRE(array.view()[0] == 42.0);

//     jl_atexit_hook(0);
// }
