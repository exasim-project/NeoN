.. _basics_RegisteredClass:

Derived class discovery at compile time
=======================================

The ``RuntimeSelectionFactory`` class is a template class that allows to register derived classes into a map. This allows to instaniate derived classes from the base class by its name.  This mechanisem is similar to OpenFOAMs runtime selection mechanism. The classes are registered at compile time via static member initialization. Additional explanation can be found at: `here <https://stackoverflow.com/questions/52354538/derived-class-discovery-at-compile-time>`_ and  `here <https://stackoverflow.com/questions/10332725/how-to-automatically-register-a-class-on-creation>`_ .

This approach allows to create a plugin architecture where now a derived class can be loaded at runtime. Additionally, it simplifies the runtime selection of a specific class. The derived classes can be created by providing its  name, as shown below:

.. code-block:: cpp

    std::unique_ptr<baseClass> derivedClass =
        baseClass::create("Name of derivedClass", "Additional arguments",...);


.. doxygenclass:: NeoFOAM::RuntimeSelectionFactory
   :members:

The ``RuntimeSelectionFactory`` class is a template class that manages the registration of the derived classes and stores the map of the registered classes. The map associates a function with a string that is used to create the derived class.


Usage
^^^^^

The following example shows how to use the ``RuntimeSelectionFactory`` and ``RuntimeSelectionFactory::Register`` to automatically register derived classes. (details see ``registerClass.cpp``).

The static function ``create`` returns a ``std::unique_ptr`` to its base class and takes an argument list specified by inside the template class ``NeoFOAM::Parameters<>``.

.. code-block:: cpp

    class BaseClass : public NeoFOAM::RuntimeSelectionFactory<BaseClass, NeoFOAM::Parameters<>>
    {

     public:

        BaseClass() {}

        static std::string name() { return "BaseClass"; }
    };

To register a class, derive from the BaseClass::Register class and pass the derived class as a template argument. The derived class must implement the static functions ``name``, ``doc``, and ``schema``.

.. code-block:: cpp

    class DerivedClass : public BaseClass::Register<DerivedClass>
    {
    public:

        DerivedClass() {}

        static std::string name() { return "DerivedClass"; }

        static std::string doc() { return "DerivedClass documentation"; }

        static std::string schema() { return "DerivedClass schema"; }
    };


After the classes have been defined,  the ``create`` function can be used to instantiate the derived classes based on the name provided.

.. code-block:: cpp

    std::unique_ptr<BaseClass> testDerived = BaseClass::create("DerivedClass");

All baseclasses are also registered in the ``NeoFOAM::BaseClassDocumentation`` map and the documentation and the schema retrieved as followed:

.. code-block:: cpp

    std::string baseClassName = "BaseClass";
    std::string derivedClass = "DerivedClass";
    NeoFOAM::BaseClassDocumentation::doc(baseClassName, derivedClass)
    NeoFOAM::BaseClassDocumentation::schema(baseClassName, derivedClass)

This mechanism should simplify the creation of tooling around NeoFOAM
