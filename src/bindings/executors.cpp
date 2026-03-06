// SPDX-FileCopyrightText: 2025 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/string.h>
#include <Kokkos_Core.hpp>

#include "NeoN/core/executor/executor.hpp"
#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace NeoN::bindings
{

void registerExecutors(nb::module_& m)
{
    nb::class_<NeoN::SerialExecutor>(
        m, "SerialExecutor", "Reference executor for serial CPU execution"
    )
        .def(nb::init<>(), "Create a serial executor")
        .def("name", &NeoN::SerialExecutor::name, "Get the executor name")
        .def("__repr__", [](const NeoN::SerialExecutor&) { return "<SerialExecutor>"; })
        .def("__str__", [](const NeoN::SerialExecutor&) { return "SerialExecutor"; })
        .def(
            "__hash__",
            [](const NeoN::SerialExecutor&) { return std::hash<std::string> {}("SerialExecutor"); }
        )
        .def(
            "__eq__", [](const NeoN::SerialExecutor&, const NeoN::SerialExecutor&) { return true; }
        );

    nb::class_<NeoN::CPUExecutor>(m, "CPUExecutor", "Executor for multicore CPU parallelization")
        .def(nb::init<>(), "Create a CPU executor")
        .def("name", &NeoN::CPUExecutor::name, "Get the executor name")
        .def("__repr__", [](const NeoN::CPUExecutor&) { return "<CPUExecutor>"; })
        .def("__str__", [](const NeoN::CPUExecutor&) { return "CPUExecutor"; })
        .def(
            "__hash__",
            [](const NeoN::CPUExecutor&) { return std::hash<std::string> {}("CPUExecutor"); }
        )
        .def("__eq__", [](const NeoN::CPUExecutor&, const NeoN::CPUExecutor&) { return true; });

    nb::class_<NeoN::GPUExecutor>(m, "GPUExecutor", "Executor for GPU offloading")
        .def(nb::init<>(), "Create a GPU executor")
        .def("name", &NeoN::GPUExecutor::name, "Get the executor name")
        .def("__repr__", [](const NeoN::GPUExecutor&) { return "<GPUExecutor>"; })
        .def("__str__", [](const NeoN::GPUExecutor&) { return "GPUExecutor"; })
        .def(
            "__hash__",
            [](const NeoN::GPUExecutor&) { return std::hash<std::string> {}("GPUExecutor"); }
        )
        .def(
            "__eq__",
            [](const NeoN::GPUExecutor&, const NeoN::GPUExecutor&)
            {
                return true; // All GPUExecutors are equal
            }
        );

    m.def(
        "executor_name",
        [](const NeoN::Executor& exec)
        { return std::visit([](auto&& e) { return e.name(); }, exec); },
        "exec"_a,
        "Get the name of an executor (works with SerialExecutor, CPUExecutor, or GPUExecutor)"
    );

    m.def(
        "executor_repr",
        [](const NeoN::Executor& exec)
        { return "<Executor: " + std::visit([](auto&& e) { return e.name(); }, exec) + ">"; },
        "exec"_a,
        "Get string representation of an executor"
    );

    m.def(
        "is_serial",
        [](const NeoN::Executor& exec)
        { return std::holds_alternative<NeoN::SerialExecutor>(exec); },
        "exec"_a,
        "Check if an executor is a SerialExecutor"
    );

    m.def(
        "is_cpu",
        [](const NeoN::Executor& exec) { return std::holds_alternative<NeoN::CPUExecutor>(exec); },
        "exec"_a,
        "Check if an executor is a CPUExecutor"
    );

    m.def(
        "is_gpu",
        [](const NeoN::Executor& exec) { return std::holds_alternative<NeoN::GPUExecutor>(exec); },
        "exec"_a,
        "Check if an executor is a GPUExecutor"
    );

    m.def(
        "gpu_available",
        []() -> bool
        {
            // Check if DefaultExecutionSpace cannot access HostSpace from device
            // This indicates the execution space runs on device (GPU)
            constexpr bool cannot_access_host_from_device =
                !Kokkos::SpaceAccessibility<NeoN::GPUExecutor::exec, Kokkos::HostSpace>::accessible;

            return cannot_access_host_from_device;
        },
        "Check if GPU acceleration is available at runtime"
    );
}

} // namespace NeoN::bindings
