// SPDX-FileCopyrightText: 2024 - 2026 NeoN authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include "NeoN/core/database/database.hpp"
#include "NeoN/core/database/collection.hpp"
#include "NeoN/core/database/document.hpp"
#include "NeoN/core/database/fieldCollection.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/volumeField.hpp"
#include "NeoN/finiteVolume/cellCentred/fields/surfaceField.hpp"

namespace NeoN::finiteVolume::cellCentred
{

// dont do dangling rereference warning here
#if defined(__GNUC__) or defined(__llvm__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdangling-reference"
#endif

class OldTimeDocument
{
public:

    OldTimeDocument(const Document& doc);

    OldTimeDocument(
        std::string nextTime, std::string previousTime, std::string currentTime, int32_t level
    );

    std::string& nextTime();

    const std::string& nextTime() const;

    std::string& previousTime();

    const std::string& previousTime() const;

    std::string& currentTime();

    const std::string& currentTime() const;

    int32_t& level();

    const int32_t& level() const;

    Document& doc();

    const Document& doc() const;

    std::string id() const;

    static std::string typeName();

private:

    Document doc_;
};


class OldTimeCollection : public CollectionMixin<OldTimeDocument>
{
public:

    OldTimeCollection(Database& db, std::string name, std::string fieldCollectionName);

    bool contains(const std::string& id) const;

    bool insert(const OldTimeDocument& cc);

    std::string findNextTime(std::string id) const;

    std::string findPreviousTime(std::string id) const;

    OldTimeDocument& oldTimeDoc(const std::string& id);

    const OldTimeDocument& oldTimeDoc(const std::string& id) const;

    template<typename VectorType>
    VectorType& getOrInsert(std::string idOfNextVector)
    {
        std::string nextId = findNextTime(idOfNextVector);
        VectorCollection& fieldCollection = VectorCollection::instance(db(), fieldCollectionName_);

        if (nextId != "") // oldVector is already registered
        {
            OldTimeDocument& oldTimeDocument = oldTimeDoc(nextId);
            return fieldCollection.fieldDoc(oldTimeDocument.previousTime()).field<VectorType>();
        }
        VectorDocument& fieldDoc = fieldCollection.fieldDoc(idOfNextVector);

        std::string oldTimeName = fieldDoc.field<VectorType>().name + "_0";
        VectorType& oldVector =
            fieldCollection.registerVector<VectorType>(CreateFromExistingVector<VectorType> {
                .name = oldTimeName,
                .field = fieldDoc.field<VectorType>(),
                .timeIndex = fieldDoc.timeIndex() - 1,
                .iterationIndex = fieldDoc.iterationIndex(),
                .subCycleIndex = fieldDoc.subCycleIndex()
            });
        OldTimeDocument oldTimeDocument(fieldDoc.field<VectorType>().key, oldVector.key, "", 0);
        setCurrentVectorAndLevel(oldTimeDocument);
        insert(oldTimeDocument);
        return oldVector;
    }

    template<typename VectorType>
    const VectorType& get(std::string idOfNextVector) const
    {
        std::string nextId = findNextTime(idOfNextVector);

        // NOTE this triggers a false positive dangling reference warning
        // on gcc <= 15
        const VectorCollection& fieldCollection =
            VectorCollection::instance(db(), fieldCollectionName_);

        if (nextId != "") // oldVector has to be registered
        {
            const OldTimeDocument& oldTimeDocument = oldTimeDoc(nextId);
            return fieldCollection.fieldDoc(oldTimeDocument.previousTime()).field<VectorType>();
        }
        else
        {
            // TODO replace with NF_THROW
            NF_ERROR_EXIT("Old field not found");
        }
    }

    static OldTimeCollection&
    instance(Database& db, std::string name, std::string fieldCollectionName);

    static const OldTimeCollection& instance(const Database& db, std::string name);

    static OldTimeCollection& instance(VectorCollection& fieldCollection);

    static const OldTimeCollection& instance(const VectorCollection& fieldCollection);

private:

    /** */
    void setCurrentVectorAndLevel(OldTimeDocument& oldTimeDoc);

    std::string fieldCollectionName_;
};

/**
 * @brief Retrieves the old time field of a given field.
 *
 * This function retrieves the old time field of a given field
 *
 * @param field The field to retrieve the old time field from.
 * @return The old time field.
 */
template<typename VectorType>
VectorType& oldTime(VectorType& field)
{
    VectorCollection& fieldCollection = VectorCollection::instance(field);
    OldTimeCollection& oldTimeCollection = OldTimeCollection::instance(fieldCollection);
    return oldTimeCollection.getOrInsert<VectorType>(field.key);
}

/**
 * @brief Retrieves the old time field of a given field (const version).
 *
 * This function retrieves the old time field of a given field
 *
 * @param field The field to retrieve the old time field from.
 * @return The old time field.
 */
template<typename VectorType>
const VectorType& oldTime(const VectorType& field)
{
    const VectorCollection& fieldCollection = VectorCollection::instance(field);
    const OldTimeCollection& oldTimeCollection = OldTimeCollection::instance(fieldCollection);
    return oldTimeCollection.get<VectorType>(field.key);
}

/**
 * @brief Helper function to retrieve the history depth of a field in oldTimeCollection.
 *
 * @param field The field to retrieve the old time field from.
 * @return History depth of the old time field.
 */
template<typename VectorType>
inline int oldTimeLevel(const VectorType& field)
{
    const auto& fieldCollection = VectorCollection::instance(field);
    const auto& oldTimeCollection = OldTimeCollection::instance(fieldCollection);

    int level = 0;
    std::string currentId = field.key;

    while (true)
    {
        std::string nextId = oldTimeCollection.findNextTime(currentId);
        if (nextId.empty())
        {
            return level;
        }

        ++level;
        currentId = oldTimeCollection.oldTimeDoc(nextId).previousTime();
    }
}

/**
 * @brief Rotate time levels for a field.
 * Meaning of `level` (computed via oldTimeLevel(field)):
 *     level == 0 : no history exists yet
 *     level == 1 : φ^{n}   exists (one oldTime buffer)
 *     level >= 2 : φ^{n} and φ^{n-1} exist (old and oldOld)
 *
 * Startup sequence when rotate() is called at the BEGINNING of each time step:
 *
 * Initial state (t = t0, before first rotate):
 *     field          = φ^{0}
 *     no oldTime buffers exist
 *
 * First rotateOldTimes() call (entering t1):
 *     - level == 0
 *     - φ^{n} buffer is allocated via oldTime(field)
 *     - data rotated: φ^{n} ← φ^{0}
 *     → history depth = 1  (BDF1 startup)
 *
 *   Second rotateOldTimes() call (entering t2):
 *     - level == 1
 *     - φ^{n-1} buffer is allocated via oldTime(oldTime(field))
 *     - data rotated: φ^{n-1} ← φ^{n} = φ^{0}, φ^{n} ← φ^{1}
 *     → history depth = 2  (BDF2 becomes active)
 *
 *   Subsequent rotate() calls:
 *     - level >= 2
 *     - buffers reused (no further allocation)
 *     - data rotated each step: φ^{n-1} ← φ^{n}, φ^{n} ← φ^{current}
 *
 * Only two historical states are kept (φ^{n}, φ^{n-1}).
 *
 * @param field The field to retrieve the old time field from.
 */
template<typename VectorType>
void rotateOldTimes(VectorType& field);

} // namespace NeoN

#if defined(__GNUC__) or defined(__llvm__)
#pragma GCC diagnostic pop
#endif
