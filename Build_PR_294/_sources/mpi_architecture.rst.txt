.. mpi_architecture:

MPI Architecture
================

Background
----------

Virtually all large scientific and engineering codes are too large to be practically usable with a single 'computing unit'. Most problems are broken down into smaller parts and distributed across several 'computing units'. In many cases these 'computing elements' do not share the same memory addressing space, or distributed memory architecture. This introduces the need to communicate data and synchronize the solution procedure across these 'computing elements'. ``NeoN`` uses `MPI <https://en.wikipedia.org/wiki/Message_Passing_Interface>`_ to achieve this, with each 'computing unit' referred to as an ``MPI rank`` (``rank``). Note: for shared memory architecture (including GPU computing), see `Executor <https://exasim-project.com/NeoN/latest/basics/executor.html>`_. Two fundamental problems need to be solved:

1. How should the global computation be partitioned and distributed across ``ranks``?.
2. What data needs to be communicated between which ``ranks``?.

For scalable solutions, it's crucial to 'mask' communication costs. Data communication should occur in parallel with the main computation to avoid holding it up and to reduce overheads. Broadly this is achieved by non-blocking communication between ``ranks`` in conjunction with the minimization of the frequency and size of the communication. Since the communication architecture informs the partitioning, we will discuss this first.

Communication
-------------

MPI Wrapping
^^^^^^^^^^^^

The majority of ``MPI`` operators are brought into ``NeoN`` in the ``operators.hpp`` file. The purpose of this file is to wrap ``MPI`` functions such that they work more seamlessly with ``NeoN`` data types, and also to supply typical defaults. For example the ``MPI_Allreduce`` function is wrapped:

.. code-block:: c++

    template<typename valueType>
    void reduceAllScalar(valueType* value, const ReduceOp op, MPI_Comm comm)
    {
        MPI_Allreduce(
            MPI_IN_PLACE, reinterpret_cast<void*>(value), 1, getType<valueType>(), getOp(op), comm
        );
    }

such that scalar reduction size is handled automatically. In addition to the wrapped operators, there is the ``MPI`` environment which is located in the ``environment.hpp`` file. Contained within are two classes ``MPIInit`` and ``MPIEnvironment``.  The former is a simple RAII class that initializes and finalizes (in the destructor) the ``MPI`` environment, thus a typically program using ``NeoN`` would start by calling the MPIInit constructor.

.. code-block:: c++

    #include "NeoN/core/mpi/environment.hpp"

    int main(int argc, char** argv)
    {
        NeoN::mpi::MPIInit mpi(argc, argv);

        // main solver

        // MPI_finalize is called in the destructor
    }

.. note::
    Since we support ``OpenMP`` (through ``Kokkos``), we need to ensure threading support is available when we start. This is checked in ``MPIInit``'s constructor.

Once started, the ``MPIEnvironment`` class is used to manage the MPI communicator, ``MPI_rank``, and ``MPI_size``. Since the class will populate the communicator with ``MPI_COMM_WORLD`` and initialize the rank and size, the class can be constructed from anywhere in the code.

.. note::
    In the future, it is intended to use this class to manage multiple MPI communicators which are derived from the splitting of an existing communicator. By then, the above will no longer apply and the instance of ``MPIEnvironment`` will need to be parsed.

Global Communication
^^^^^^^^^^^^^^^^^^^^

With the above in place, global communication (i.e. communication between all ``ranks`` on a ``MPI_Communicator``) can be achieved by using the environment and operators.

.. code-block:: c++

    #include "NeoN/core/mpi/environment.hpp"
    #include "NeoN/core/mpi/operators.hpp"

    int main(int argc, char** argv)
    {
        NeoN::mpi::MPIInit mpi(argc, argv);
        NeoN::mpi::MPIEnvironment mpiEnv;

        double value = 1.0;
        NeoN::mpi::reduceAllScalar(&value, NeoN::mpi::ReduceOp::SUM, mpiEnv.comm());

        if(mpiEnv.rank() == 0)
            std::cout<<"Value "<<value<<std::endl; // result is number of ranks.
    }

Point-to-Point Communication
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For simplicity, this section focuses on the approach for two ranks to communicate with each other, specifically using non-blocking communication for field data synchronization.

To begin, the reader is reminded of 'communication terminology': simplex, half-duplex, and full-duplex. Simplex communication is one-way, from sender to receiver or vice versa. Half-duplex allows two-way communication but only in one direction at a time. Full-duplex enables two-way communication simultaneously in both directions.

To facilitate communication between two or more ranks, a half-duplex buffer is introduced, namely the ``HalfDuplexCommBuffer``, which is responsible for non-blocking sending to/receiving from different ranks and into member data buffers. To generalize the buffer for different data types, ``type-punning`` is used and as such the actual data which is transferred is always of type ``char``. Further, since memory allocation is relatively expensive the buffer is never sized down. While the buffer memory is laid out continuously it is accessed on a per ``rank`` basis, which is indexed from 0 to the size for the communicated data. It is therefore required to have some map between a cell's buffer position index and its data container (typically a ``Vector`` of some kind) index. The construction of this map is part of the partitioning problem, and not the responsibility of the buffer.

.. note::
    The ``HalfDuplexCommBuffer`` duplex buffer has some guard rails in to ensure that once communication has started, various operations are no-longer possible until it is finished.

To achieve full-duplex communication, two half-duplex buffers are combined to form the ``FullDuplexCommBuffer``. The process for two way communication is then broken down into the following steps:

1. Initialize the communication, using a name and data type. This flags the buffer as a used resource.
2. Load the buffer with data to send.
3. Start the communication.
4. Do other work to mask the communication.
5. Wait for the communication to finish.
6. Unload the buffer with the received data.
7. Finalize the communication, releasing (de-flags) the buffer.

The full communication between two ranks is thus given below:

.. code-block:: c++

    #include <unordered_map>
    #include <vector>
    #include "NeoN/core/mpi/environment.hpp"
    #include "NeoN/core/mpi/operators.hpp"
    #include "NeoN/core/mpi/comm_buffer.hpp"

    int main(int argc, char** argv)
    {
        NeoN::mpi::MPIInit mpi(argc, argv);
        NeoN::mpi::MPIEnvironment mpiEnv;

        // create the buffers
        std::vector<std::size_t> sendSize;      // per rank communication
        std::vector<std::size_t> receiveSize;   // per rank communication
        std::vector<double> allData = {1.0, 2.0, 3.0}; // the local data (could be a field or similar)
        std::unordered_map<std::size_t, std::size_t> sendMap; // assumes single rank communication
        std::unordered_map<std::size_t, std::size_t> receiveMap; // assumes single rank communication

        // ...
        // populate above data
        // ...

        NeoN::mpi::FullDuplexCommBuffer buffer(mpiEnv, sendSize, receiveSize);

        // Obtain the buffer.
        buffer.initComm<double>("test_communication");

        // load the send buffer
        const int commRank = mpiEnv.Rank() ? 1 : 0;
        auto sendBuffer = buffer.getSendBuffer<double>(commRank); // span returned.
        sendBuffer[0] = allData[sendMap[0]];

        // start the non-blocking communication
        buffer.startComm();

        // ...
        // do other work
        // ...

        // wait for the communication to finish
        buffer.waitComplete();

        // unload the receive buffer
        auto receiveBuffer = buffer.getReceiveBuffer<double>(commRank); // span returned.
        allData[receiveMap[0]] = receiveBuffer[0];

        // finalize the communication, releasing the buffer
        buffer.finaliseComm();
    }

.. note::
    The copying to and from the buffers does introduce an overhead, which could later be removed by using 'inplace' communication. This remains an open point.

.. note::
    In the future it is aimed to have dead-lock detection, to prevent program hangs when developing MPI based algorithms.

Vector Synchronization
^^^^^^^^^^^^^^^^^^^^^

The focus now shifts to the actual process of synchronizing a global field between all its partitioned parts. In each ``rank`` there is some overlap of cells (i.e. cells which are present in more than one ``rank``), which is dictated by the stencil size. If these shared cell have a missing neighbor cell in a local partition they are termed ``halo cells``. A ``halo cell`` does not have enough geometric and/or field information to be able to calculate the correct result and therefore must receive the result from another rank.

In the above there is no reason for the ``halo cells`` to be nicely ordered, for example to start at field index 0 and end at 10. Therefore we need some map between the ``halo cell`` index in our mesh and our data buffers in the ``FullDuplexCommBuffer``, for each ``rank``. This map is stored in the ``RankSimplexCommMap`` which stores for each ``rank`` which buffer position maps to which ``halo cell`` in the mesh. To facilitate full duplex communication both a send and receive ``RankSimplexCommMap`` is needed.

Arriving finally at the ``Communicator``. Its role is now defined to manage the non-blocking synchronization of a field for a given communication pathway set. The user should, for each communicate point in code, provide a unique string key to identify the communication, see below is an example.

It is worth noting that there may be more than one field being synchronized at any give time. However, the communication pathways contained within the send and receive ``RankSimplexCommMap`` remains the same. Thus the ``Communicator`` (may) consists of a multiple of communication buffers and a single ``RankSimplexCommMap``. This scaling is provided automatically.

.. code-block::c++

    mpi::MPIEnvironment MPIEnviron;
    Communicator comm;

    Vector<int> field(SerialExecutor());

    // ...
    // Size and populate field data.
    // ...

    // Set up buffer to local map
    RankSimplexCommMap rankSendMap(MPIEnviron.sizeRank());
    RankSimplexCommMap rankReceiveMap(MPIEnviron.sizeRank());

    // ...
    // Set up of send/receive maps per rank.
    // ...

    // Set up a communicatory.
    comm = Communicator(MPIEnviron, rankSendMap, rankReceiveMap);

    std::string loc =
        std::source_location::current().file_name() + std::source_location::current().line(); // used to identify the communication
    comm.startComm(field, loc);
    comm.isComplete(loc);
    comm.finaliseComm(field, loc);

.. note::
    If the file line and number are used as communication key names they can allow for helpful debug messages where ``MPI`` communication throws an error.

In the above of course the logic would be situated in a solution loop, and the calls would not be made sequential as this would lead to blocking communication.

Partitioning
------------

The purpose of partitioning is to divide the global computation into smaller parts that can be solved in parallel, and essentially to distribute the computation across the ``ranks``.

Currently there is no formal partitioning system in ``NeoN``, however it is assumed that all communication is done on the ``MPI World`` communicator. This is to be updated in the future, together with dynamic load balancing.


Future Work
-----------

1. Allow ``MPI Communicators`` to be split, allowing for more complex partitioning of the computation.
2. GPU support.
3. Mesh partitioning
4. dead-lock detection.
5. Implement dynamic load balancing.
6. Replace, where possible, std containers with ``NeoN`` and/or ``Kokkos`` containers.
7. Performance metrics
