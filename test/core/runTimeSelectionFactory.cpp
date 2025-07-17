// SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
//
// SPDX-License-Identifier: MIT

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_approx.hpp>

// #include "NeoN/NeoN.hpp"
#include "NeoN/core/runtimeSelectionFactory.hpp"

class BaseClass : public NeoN::RuntimeSelectionFactory<BaseClass, NeoN::Parameters<>>
{

public:

    BaseClass() {}

    virtual int doSomeThing() = 0;

    static std::string name() { return "BaseClass"; }
};

class DerivedClass : public BaseClass::Register<DerivedClass>
{
public:

    DerivedClass() {}

    virtual int doSomeThing() override { return 1; }

    static std::string name() { return "DerivedClass"; }

    static std::string doc() { return "DerivedClass documentation"; }

    static std::string schema() { return "DerivedClass schema"; }
};

template<typename T>
class BaseClass2 : public NeoN::RuntimeSelectionFactory<BaseClass2<T>, NeoN::Parameters<>>
{

public:

    BaseClass2() {}

    virtual T doSomeThing(T in) = 0;

    static std::string name() { return "BaseClass2"; }
};

template<typename T>
class DerivedClass2 : public BaseClass2<T>::template Register<DerivedClass2<T>>
{
public:

    DerivedClass2() {}

    virtual T doSomeThing(T in) override { return 2 * in; }

    static std::string name() { return "DerivedClass2"; }

    static std::string doc() { return "DerivedClass2 documentation"; }

    static std::string schema() { return "DerivedClass2 schema"; }
};

// in case of registered template classes templates need to be instantiated explicitly
template class DerivedClass2<float>;
template class DerivedClass2<int>;

TEST_CASE("RunTimeSelectionFactory")
{
    std::cout << "Table size: " << NeoN::BaseClassDocumentation::docTable().size() << std::endl;

    SECTION("classes are registered")
    {
        CHECK(NeoN::BaseClassDocumentation::docTable().size() == 2);
        for (const auto& it : NeoN::BaseClassDocumentation::docTable())
        {
            std::string baseClassName = it.first;
            std::cout << "baseClassName " << baseClassName << std::endl;
            auto entries = NeoN::BaseClassDocumentation::entries(baseClassName);
            for (const auto& derivedClass : entries)
            {
                std::cout << "   - " << derivedClass << std::endl;
                std::cout << "     doc: "
                          << NeoN::BaseClassDocumentation::doc(baseClassName, derivedClass)
                          << std::endl;
                std::cout << "     schema: "
                          << NeoN::BaseClassDocumentation::schema(baseClassName, derivedClass)
                          << std::endl;
                CHECK(!NeoN::BaseClassDocumentation::doc(baseClassName, derivedClass).empty());
                CHECK(!NeoN::BaseClassDocumentation::schema(baseClassName, derivedClass).empty());
            }
        }
    }
    SECTION("classes can be constructed")
    {

        auto derivedA = BaseClass::create("DerivedClass");
        REQUIRE(derivedA->doSomeThing() == 1);

        auto derivedB = BaseClass2<float>::create("DerivedClass2");
        REQUIRE(derivedB->doSomeThing(2.5) == 5.0);
    }
}
