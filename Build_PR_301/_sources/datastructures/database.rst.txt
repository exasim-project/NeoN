.. _basics_Database:

Database
========

OpenFOAM stores objects in the ``objectRegistry`` and they can be accessed by their name and type.
In modern software development, best practices emphasize modularity, testability, and loosely coupled components.
While the ``objectRegistry`` of OpenFOAM offers convenience and simple approach for managing objects, its inherent design conflicts with these principles by:

- Encouraging reliance on a central container.
- Making the codebase less flexible for testing, refactoring, or adopting new architectural patterns.
- Introducing potential performance bottlenecks and debugging challenges.

In contrast, NeoFOAM adopts a document-based database approach, where data is stored as a collection of documents.
Each document consists of a pair of strings and values, validated using a custom validator function to ensure data integrity.
The validation ensures that the data stored in the documents adheres to predefined rules and formats, preventing errors and inconsistencies in the database.
The database is not tightly coupled to the ``fvMesh`` or ``Time`` classes, making it more flexible and easier to test.

The following class diagram illustrates the relationships between the ``Database``, ``Collection``, and ``Document`` components:

.. mermaid::

    classDiagram

        class Database{
            -std::unordermap&lt;std::string, Collection> data
        }

        class CollectionA{
            +std::vector&lt;std::string> find(std::function predicate)
            -std::unordermap&lt;std::string, Document> data
        }

        class CollectionB{
            +std::vector&lt;std::string> find(std::function predicate)
            -std::unordermap&lt;std::string, Document> data
        }

        class CollectionC["..."]{
            +std::vector&lt;std::string> find(std::function predicate)
            -std::unordermap&lt;std::string, Document> data
        }

        class DocumentA{
            +validate() bool
            -id_ : std::string
            -validator : std::function
            -dict : Dictionary
        }

        class DocumentB{
            +validate() bool
            -id_ : std::string
            -validator : std::function
            -dict : Dictionary
        }

        class DocumentC{
            +validate() bool
            -id_ : std::string
            -validator : std::function
            -dict : Dictionary
        }

        class DocumentD{
            +validate() bool
            -id_ : std::string
            -validator : std::function
            -dict : Dictionary
        }

        class DocumentE["..."]{
            +validate() bool
            -id_ : std::string
            -validator : std::function
            -dict : Dictionary
        }

        class DocumentF["..."]{
            +validate() bool
            -id_ : std::string
            -validator : std::function
            -dict : Dictionary
        }


        Database <-- CollectionA
        Database <-- CollectionB
        Database <-- CollectionC

        CollectionA <-- DocumentA
        CollectionA <-- DocumentB
        CollectionA <-- DocumentE
        CollectionB <-- DocumentC
        CollectionB <-- DocumentD
        CollectionB <-- DocumentF

A database can have 0 to N collections and each collection can have 0 to N documents.
At the lowest level is the ``Document`` class, which is a Dictionary with an ID and a validator function to ensure data integrity.
The Document container is similar to a python dictionary using key-value pairs and can be used to store any type of data.
The following code snippet shows how to create a document and access its values:

.. sourcecode:: cpp

    NeoFOAM::Document doc({{"key1", std::string("value1")}, {"key2", 2.0}});
        REQUIRE(doc.keys().size() == 3);
        REQUIRE(doc.id().substr(0, 4) == "doc_");
        REQUIRE(doc.get<std::string>("key1") == "value1");
        REQUIRE(doc.get<double>("key2") == 2.0);
    };

``NeoFOAM::Document`` mainly extends the ``Dictionary`` class and offers the possibility to validate the data stored in the document.
The following code snippet shows how to create a document with a custom validator function:

.. sourcecode:: cpp

    auto validator = [](const NeoFOAM::Dictionary& dict)
    { return dict.contains("key1") && dict.contains("key2"); };

    NeoFOAM::Document doc({{"key1", std::string("value1")}, {"key2", 2.0}}, validator);
    REQUIRE_NOTHROW(doc.validate());

As stated earlier, the Documents are stored as part of a Collection which itself is stored in the central database as shown in the class diagram above.
To enable custom functionalities the developer can create a custom collection that provides additional functionality to the document.
This is done in the FieldCollection class that is used to store the field data in the database.



FieldCollection
---------------

The ``FieldCollection`` stores all fields and provides additional functionality which is stored in a document.
The ``FieldDocument`` keeps a reference to field as a ``std::any`` and stores additional metadata like the time index, iteration index, and subcycle index.
The following code snippet shows how to create a ``FieldDocument``:

.. sourcecode:: cpp

    template<class FieldType>
    FieldDocument(
        const FieldType& field,
        std::size_t timeIndex,
        std::size_t iterationIndex,
        std::int64_t subCycleIndex
    )
        : doc_(
              Document(
                  {{"name", field.name},
                   {"timeIndex", timeIndex},
                   {"iterationIndex", iterationIndex},
                   {"subCycleIndex", subCycleIndex},
                   {"field", field}}
              ),
              validateFieldDoc
          )
    {}

The ``FieldDocument`` stores its data in a ``Document`` and provide the appropriate getters and setters for the data.
The user will most likely not directly create FieldDocument but use the ``FieldCollection`` to register Field.

.. sourcecode:: cpp

    fvcc::FieldCollection& fieldCollection =
        fvcc::FieldCollection::instance(db, "newTestFieldCollection");

    fvcc::VolumeField<NeoFOAM::scalar>& T =
        fieldCollection.registerField<fvcc::VolumeField<NeoFOAM::scalar>>(CreateField {
            .name = "T", .mesh = mesh, .timeIndex = 1, .iterationIndex = 1, .subCycleIndex = 1
        });

The ``FieldCollection`` has a static instance methods that returns the instance of the ``FieldCollection`` or creates a new one if it doesn't exist.
registerField method is used to register fields and expects a CreateFunction that returns a FieldDocument and expects a database as an argument.
The createFunction could look as followed:

.. sourcecode:: cpp

    struct CreateField
    {
        std::string name;
        NeoFOAM::UnstructuredMesh mesh;
        std::size_t timeIndex = 0;
        std::size_t iterationIndex = 0;
        std::int64_t subCycleIndex = 0;
        NeoFOAM::Document operator()(NeoFOAM::Database& db)
        {
            std::vector<fvcc::VolumeBoundary<NeoFOAM::scalar>> bcs {};
            for (auto patchi : std::vector<size_t> {0, 1, 2, 3})
            {
                NeoFOAM::Dictionary dict;
                dict.insert("type", std::string("fixedValue"));
                dict.insert("fixedValue", 2.0);
                bcs.push_back(fvcc::VolumeBoundary<NeoFOAM::scalar>(mesh, dict, patchi));
            }
            NeoFOAM::Field internalField =
                NeoFOAM::Field<NeoFOAM::scalar>(mesh.exec(), mesh.nCells(), 1.0);
            fvcc::VolumeField<NeoFOAM::scalar> vf(
                mesh.exec(), name, mesh, internalField, bcs, db, "", ""
            );
            return NeoFOAM::Document(
                {{"name", vf.name},
                {"timeIndex", timeIndex},
                {"iterationIndex", iterationIndex},
                {"subCycleIndex", subCycleIndex},
                {"field", vf}},
                fvcc::validateFieldDoc
            );
        }
    };

This design can be easily extended to create and read fields.
The ``FieldCollection`` allows us to access and find fields by their name, time index, iteration index, or subcycle index.

.. sourcecode:: cpp

    fvcc::FieldCollection& fieldCollection =
        fvcc::FieldCollection::instance(db, "newTestFieldCollection");

    auto resName = fieldCollection.find([](const NeoFOAM::Document& doc)
                                    { return doc.get<std::string>("name") == "T"; });

    REQUIRE(resName.size() == 1);
    const auto& fieldDoc = fieldCollection.fieldDoc(resName[0]);
    const auto& constVolField = fieldDoc.field<fvcc::VolumeField<NeoFOAM::scalar>>();

Query of document in a collection
---------------------------------

The ``Collection`` class provides a find method that allows the user to query the documents in the collection.

.. sourcecode:: cpp

    std::vector<std::string> keys = fieldCollection.find(
        [](const Document& doc)
        {
            return doc.get<std::string>("name") == "someName";
                && doc.get<std::size_t>("someValue") == 42.0;
        }
    );

The developer can provide a lambda function that returns a boolean value to filter the documents in the collection.
The find function returns a vector of keys that match the query.

Adding a new collection and documents to the database
-----------------------------------------------------

The database supports extensibility through a type-erased interface for collections.
This design allows developers to create and manage collections of custom documents with minimal coupling.
A collection provides access to its stored documents, and the only requirement for custom documents is that they extend the base Document class and implement the necessary functionality.

Creating a Custom Document
^^^^^^^^^^^^^^^^^^^^^^^^^^

Custom documents extend the Document class and add domain-specific functionality. For example:


.. sourcecode:: cpp

    class CustomDocument
    {
    public:

        CustomDocument(
            const std::string& name,
            const double& testValue
        )
            : doc_(
                NeoFOAM::Document(
                    {
                        {"name", name},
                        {"testValue", testValue}
                    }
                    , validateCustomDoc
                )
            )
        {}

        std::string& name() { return doc_.get<std::string>("name"); }

        const std::string& name() const { return doc_.get<std::string>("name"); }

        double testValue() const { return doc_.get<double>("testValue"); }

        double& testValue() { return doc_.get<double>("testValue"); }


        NeoFOAM::Document& doc() { return doc_; }

        const NeoFOAM::Document& doc() const { return doc_; }

        std::string id() const { return doc_.id(); }

        static std::string typeName() { return "CustomDocument"; }

    private:

        NeoFOAM::Document doc_;
    };

Here, the CustomDocument class:

    1. Wraps the base Document with its own fields (name and testValue) and validates it.
    2. Provides accessor functions for these variables

The Document class can be easily wrapped with a custom class to provide additional functionality as shown below.

Storing Custom Documents in a Custom Collection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Collections manage a set of user-defined documents and provide operations to query and modify them.
The CollectionMixin template simplifies creating custom collections for user-defined documents. For instance:


.. sourcecode:: cpp

    class CustomCollection : public NeoFOAM::CollectionMixin<CustomDocument>
    {
    public:

        CustomCollection(NeoFOAM::Database& db, std::string name)
            : NeoFOAM::CollectionMixin<CustomDocument>(db, name)
        {}

        bool contains(const std::string& id) const { return docs_.contains(id); }

        bool insert(const CustomDocument& cc)
        {
            std::string id = cc.id();
            if (contains(id))
            {
                return false;
            }
            docs_.emplace(id, cc);
            return true;
        }

        static CustomCollection& instance(NeoFOAM::Database& db, std::string name)
        {
            NeoFOAM::Collection& col = db.insert(name, CustomCollection(db, name));
            return col.as<CustomCollection>();
        }
    };

In this example:

    1. CustomCollection inherits from CollectionMixin<CustomDocument>, which already provides the necessary functionality for managing custom documents.
    2. instance is a static method that returns the instance of the CustomCollection or creates a new one if it doesn't exist.

This design allows developers to create custom collections with minimal boilerplate code and focus on the domain-specific functionality of the documents.
