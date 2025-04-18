// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 AMR Wind Authors
// SPDX-FileCopyrightText: 2023-2024 NeoN authors
// ##############################################################################
// # Original design taken from amr wind                                        #
// # from here                                                                  #
// # https://github.com/Exawind/amr-wind/blob/v2.1.0/amr-wind/core/Factory.H    #
// ##############################################################################
// its quite tricky for multiple compilers that bool REGISTERED gets initialized
// the static_assert helps to register the class
// https://stackoverflow.com/questions/6420985/
// how-to-force-a-static-member-to-be-initialized?noredirect=1&lq=1
#pragma once

#include <memory>
#include <unordered_map>
#include <iostream>
#include <functional>

#include "error.hpp"

namespace NeoN
{

/**
 * @class BaseClassData
 * @brief Represents the data for a base class.
 *
 * This class holds the information related to a base class, including documentation,
 * schema, and entries.
 */
class BaseClassData
{
public:

    std::function<std::string(const std::string&)>
        doc; /**< Function to retrieve the documentation for a specific entry. */
    std::function<std::string(const std::string&)>
        schema; /**< Function to retrieve the schema for a specific entry. */
    std::function<std::vector<std::string>()>
        entries; /**< Function to retrieve the list of entries. */
};

/**
 * @class BaseClassDocumentation
 * @brief Provides a mechanism for registering and retrieving documentation for base and derived
 * classes.
 *
 * The BaseClassDocumentation class allows users to register documentation for base classes and
 * retrieve documentation for derived classes based on the registered information. It also provides
 * methods to retrieve the schema and entries associated with a base class.
 */
class BaseClassDocumentation
{
public:

    using LookupTable = std::unordered_map<std::string, BaseClassData>;

    static void registerClass(std::string name, BaseClassData data)
    {
        // if not already registered
        docTable()[name] = data;
    }

    /**
     * Returns the documentation for a derived class based on the base class name.
     *
     * @param baseClassName The name of the base class.
     * @param derivedClassName The name of the derived class.
     * @return The documentation for the derived class.
     */
    static std::string doc(const std::string& baseClassName, const std::string& derivedClassName)
    {
        return docTable().at(baseClassName).doc(derivedClassName);
    }

    /**
     * Returns the schema of the derived class based on the base class name and derived class name.
     *
     * @param baseClassName The name of the base class.
     * @param derivedClassName The name of the derived class.
     * @return The schema of the derived class.
     */
    static std::string schema(const std::string& baseClassName, const std::string& derivedClassName)
    {
        // get the schema of the derived class
        return docTable().at(baseClassName).schema(derivedClassName);
    }

    /**
     * Returns a vector of strings representing the entries for a given base class name.
     *
     * @param baseClassName The name of the base class.
     * @return A vector of strings representing the entries.
     */
    static std::vector<std::string> entries(const std::string& baseClassName)
    {
        return docTable().at(baseClassName).entries();
    }

    static LookupTable& docTable()
    {
        static LookupTable tbl;
        return tbl;
    }
};


/**
 * @brief Template struct for registering documentation of a base class.
 *
 * This struct is used to register the documentation of a base class. It provides a static function
 * `init()` that registers the class documentation by calling
 * `BaseClassDocumentation::registerClass()`. The registration includes the class name,
 * documentation, schema, and entries.
 *
 * @tparam baseClass The base class for which the documentation is being registered.
 */
template<class baseClass>
struct RegisterDocumentation
{
    RegisterDocumentation()
    {
        // avoid unused variable warning
        // is required to instantiate the static variable and with it the registration
        (void)REGISTERED;
    }

    /**
     * @brief Static function to initialize the registration of the class documentation.
     *
     * This function is called to register the class documentation by calling
     * `BaseClassDocumentation::registerClass()`. It includes the class name, documentation, schema,
     * and entries.
     *
     * @return Always returns true.
     */
    static bool init()
    {
        BaseClassData data = {baseClass::doc, baseClass::schema, baseClass::entries};
        BaseClassDocumentation::registerClass(baseClass::name(), data);
        return true;
    }

    static bool REGISTERED; ///< Static variable used to trigger the registration of the class
                            ///< documentation.
#ifdef _MSC_VER
    static_assert((bool)&REGISTERED);
#endif
};

// Initialize the static variable and register the class
template<class baseClass>
bool RegisterDocumentation<baseClass>::REGISTERED = RegisterDocumentation<baseClass>::init();

/**
 * @brief Class representing the documentation for a derived class.
 *
 * This class stores the documentation and schema information for a derived class.
 */
class DerivedClassDocumentation
{
public:

    std::function<std::string()> doc;
    std::function<std::string()> schema;
};

// Parameters helper type
template<typename... Args>
struct Parameters
{
};

// Primary template declaration
template<typename Base, typename Params>
class RuntimeSelectionFactory;

// Partial specialization for Parameters
/**
 * @class RuntimeSelectionFactory
 * @brief A factory class for runtime selection of derived classes.
 *
 * The `RuntimeSelectionFactory` class provides a mechanism for creating and managing instances of
 * derived classes at runtime. It allows registration of derived classes and provides methods for
 * creating instances, accessing documentation and schemas, and retrieving a list of available
 * options.
 *
 * @tparam Base The base class from which the derived classes inherit.
 * @tparam Parameters The template parameters for the derived classes.
 */
template<typename Base, typename... Args>
class RuntimeSelectionFactory<Base, Parameters<Args...>> : public RegisterDocumentation<Base>
{
public:

    friend Base;

    using CreatorFunc = std::function<std::unique_ptr<Base>(Args...)>;
    using LookupTable = std::unordered_map<std::string, CreatorFunc>;
    using ClassDocTable = std::unordered_map<std::string, DerivedClassDocumentation>;

    /**
     * Returns the documentation of the derived class.
     *
     * @param derivedClassName The name of the derived class.
     * @return The documentation of the derived class.
     */
    static std::string doc(const std::string& derivedClassName)
    {
        // get the documentation of the derived class
        return docTable().at(derivedClassName).doc();
    }

    /**
     * Returns the schema of the derived class.
     *
     * @param derivedClassName The name of the derived class.
     * @return The schema of the derived class.
     */
    static std::string schema(const std::string& derivedClassName)
    {
        // get the schema of the derived class
        return docTable().at(derivedClassName).schema();
    }

    /**
     * @brief Get a vector of all entries in the runtime selection factory.
     *
     * This function retrieves all entries in the runtime selection factory and returns them as a
     * vector of strings.
     *
     * @return A vector of strings containing all entries in the runtime selection factory.
     */
    static std::vector<std::string> entries()
    {
        std::vector<std::string> entries;
        for (const auto& it : table())
        {
            entries.push_back(it.first);
        }
        return entries;
    }


    /**
     * @brief Creates an instance of a derived class based on the provided key.
     *
     * This function creates an instance of a derived class based on the provided key.
     * The key is used to look up a factory function in a table, and the factory function
     *  is then called with the provided arguments to create the instance.
     *
     * @param key The key used to look up the factory function in the table.
     * @param args The arguments to be forwarded to the factory function.
     * @return A unique pointer to the created instance.
     *
     * @throws std::out_of_range if the provided key does not exist in the table.
     */
    static std::unique_ptr<Base> create(const std::string& key, Args... args)
    {
        keyExistsOrError(key);
        auto ptr = table().at(key)(std::forward<Args>(args)...);
        return ptr;
    }


    /**
     * Prints the name and size of the table, followed by each entry in the table.
     *
     * @param os The output stream to print the information to.
     */
    static void print(std::ostream& os)
    {
        const auto& tbl = table();
        os << Base::name() << " " << tbl.size() << std::endl;
        for (const auto& it : tbl)
        {
            os << " - " << it.first << std::endl;
        }
    }

    /**
     * @class RuntimeSelectionFactory::Register
     * @brief A template class for registering derived classes with a base class.
     *
     * This class provides a mechanism for registering derived classes with a base class
     * using a runtime selection factory. It also provides a way to associate documentation
     * and schema information with each derived class.
     *
     * @tparam derivedClass The derived class to be registered.
     */
    template<class derivedClass>
    class Register : public Base
    {
    public:

        using Base::Base;

        friend derivedClass;
        [[maybe_unused]] static bool REGISTERED;
#ifdef _MSC_VER
        static_assert((bool)&REGISTERED);
#endif

        /**
         * @brief Adds the derived class as a sub type.
         *
         * This function adds the derived class to the table and docTable of the runtime selection
         * factory. It is instantiated via the template below
         *
         * @tparam Args The argument types for constructing the derived class.
         * @param args The arguments for constructing the derived class.
         * @return True if the derived class was successfully added as a sub type, false otherwise.
         */
        static bool addSubType()
        {
            CreatorFunc func = [](Args... args) -> std::unique_ptr<Base> {
                return static_cast<std::unique_ptr<Base>>(new derivedClass(std::forward<Args>(args
                )...));
            };
            RuntimeSelectionFactory::table()[derivedClass::name()] = func;

            DerivedClassDocumentation childData;
            childData.doc = []() -> std::string { return derivedClass::doc(); };
            childData.schema = []() -> std::string { return derivedClass::schema(); };
            RuntimeSelectionFactory::docTable()[derivedClass::name()] = childData;

            return true;
        }

        ~Register() override
        {
            if (REGISTERED)
            {
                const auto& tbl = RuntimeSelectionFactory::table();
                const auto it = tbl.find(derivedClass::name());
                REGISTERED = (it != tbl.end());
            }
        }

#ifdef _MSC_VER
    private:

        Register() { (void)REGISTERED; }
#endif
    };

    virtual ~RuntimeSelectionFactory() = default;

    static std::size_t size() { return table().size(); }

    /**
     * @brief Returns the lookup table for runtime selection.
     *
     * This function returns a reference to the lookup table used for runtime selection.
     * The lookup table is a static object that is lazily initialized and only created once.
     *
     * @return A reference to the lookup table.
     */
    static LookupTable& table()
    {
        static LookupTable tbl;
        return tbl;
    }

    /**
     * @brief Returns the documentation table for runtime selection.
     *
     * This function returns a reference to the documentation table used for runtime selection.
     * The documentation table is a static object that is lazily initialized and only created once.
     *
     * @return A reference to the documentation table.
     */
    static ClassDocTable& docTable()
    {
        static ClassDocTable tbl;
        return tbl;
    }

private:


    /**
     * Checks if a given key exists in the table and prints an error message if it doesn't.
     *
     * @param key The key to check for existence in the table.
     */
    static void keyExistsOrError(const std::string& name)
    {
        const auto& tbl = table();
        if (tbl.find(name) == tbl.end())
        {
            auto msg = std::string {" Could not find constructor for "} + name + "\n";
            msg += "valid constructors are: \n";
            for (const auto& it : tbl)
            {
                msg += " - " + it.first + "\n";
            }
            NF_ERROR_EXIT(msg);
        }
    }

    RuntimeSelectionFactory() = default;
};

// Initialize the static variable and register the class
template<class Base, class... Args>
template<class derivedClass>
bool RuntimeSelectionFactory<Base, Parameters<Args...>>::Register<derivedClass>::REGISTERED =
    RuntimeSelectionFactory<Base, Parameters<Args...>>::template Register<derivedClass>::addSubType(
    );

}; // namespace NeoN
