// SPDX-FileCopyrightText: 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#if NF_WITH_GINKGO

#include <sstream>

#include "NeoN/linearAlgebra/ginkgo.hpp"

gko::config::pnode NeoN::la::ginkgo::parse(const Dictionary& dict)
{
    // check if an external file name is given
    if (dict.contains("configFile"))
    {

        std::string fn_str {};
        auto fn = dict["configFile"];
        if (fn.type() == typeid(std::string))
        {
            fn_str = std::any_cast<std::string>(fn);
        }
        else
        {
            auto token = std::any_cast<TokenList>(fn);
            std::stringstream s;
            for (auto i = 0; i < token.size() - 1; i++)
            {
                s << token.next<std::string>() << "/";
            }
            s << token.next<std::string>();
            fn_str = s.str();
        }

        return gko::ext::config::parse_json_file(fn_str);
    }

    auto parseData = [&](auto key)
    {
        auto parseAny = [&](auto blueprint)
        {
            using value_type = decltype(blueprint);
            if (dict[key].type() == typeid(value_type))
            {
                return gko::config::pnode(dict.get<value_type>(key));
            }
            else
            {
                return gko::config::pnode();
            }
        };

        if (auto node = parseAny(std::string()))
        {
            return node;
        }
        if (auto node = parseAny(static_cast<const char*>(nullptr)))
        {
            return node;
        }
        if (auto node = parseAny(int {}))
        {
            return node;
        }
        if (auto node = parseAny(static_cast<unsigned int>(0)))
        {
            return node;
        }
        if (auto node = parseAny(double {}))
        {
            return node;
        }
        if (auto node = parseAny(float {}))
        {
            return node;
        }

        NF_THROW("Dictionary key " + key + " has unsupported type: " + dict[key].type().name());
    };
    gko::config::pnode::map_type result;
    for (const auto& key : dict.keys())
    {
        gko::config::pnode node;
        if (dict.isDict(key))
        {
            node = parse(dict.subDict(key));
        }
        else
        {
            node = parseData(key);
        }
        result.emplace(key, node);
    }
    return gko::config::pnode {result};
}

// TODO: check if this can be replaced by Ginkgos executor mapping
std::shared_ptr<gko::Executor> NeoN::la::ginkgo::getGkoExecutor(NeoN::Executor exec)
{
    return std::visit(
        [](auto concreteExec) -> std::shared_ptr<gko::Executor>
        {
            using ExecType = std::decay_t<decltype(concreteExec)>;
            if constexpr (std::is_same_v<ExecType, NeoN::SerialExecutor>)
            {
                return gko::ReferenceExecutor::create();
            }
            else if constexpr (std::is_same_v<ExecType, NeoN::CPUExecutor>)
            {
#if defined(KOKKOS_ENABLE_OMP)
                return gko::OmpExecutor::create();
#elif defined(KOKKOS_ENABLE_THREADS)
                return gko::ReferenceExecutor::create();
#endif
            }
            else if constexpr (std::is_same_v<ExecType, NeoN::GPUExecutor>)
            {
#if defined(KOKKOS_ENABLE_CUDA)
                return gko::CudaExecutor::create(
                    Kokkos::device_id(), gko::ReferenceExecutor::create()
                );
#elif defined(KOKKOS_ENABLE_HIP)
                return gko::HipExecutor::create(
                    Kokkos::device_id(), gko::ReferenceExecutor::create()
                );
#endif
                throw std::runtime_error("No valid GPU executor mapping available");
            }
            else
            {
                throw std::runtime_error("Unsupported executor type");
            }
            return gko::ReferenceExecutor::create();
        },
        exec
    );
}

#endif
