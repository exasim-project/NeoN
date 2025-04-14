Expression
---------


The `Expression` class in NeoN holds a set of operators to express an expression and ultimately helps to formulate equations.
Its core responsibility lies in the answering of the following questions:

    - How to discretize the spatial terms?
        - In OpenFOAM this information is provided in **fvSchemes**
    - How to integrate the system in time?
        - In OpenFOAM this information is provided in **fvSchemes**
    - How to solve the system?
        - In OpenFOAM this information is provided in **fvSolution**

The main difference between NeoN and OpenFOAM is that the DSL is evaluated lazily, i.e. evaluation is not performed on construction by default.
Since, the evaluation is not tied to the construction but rather delayed, other numerical integration strategies (e.g. RK methods or even sub-stepping within an the equation) are possible.

To implement lazy evaluation, the `Expression` stores 3 vectors:

.. mermaid::

    classDiagram
        class Operator {
            +explicitOperation(...)
            +implicitOperation(...)
        }
        class DivOperator {
            +explicitOperation(...)
            +implicitOperation(...)
        }
        class TemporalOperator {
            +explicitOperation(...)
            +implicitOperation(...)
        }
        class Others["..."] {
            +explicitOperation(...)
            +implicitOperation(...)
        }
        class Expression {
            +temporalTerms_: vector~Operator~
            +implicitTerms_: vector~Operator~
            +explicitTerms_: vector~Operator~
        }
        Operator <|-- DivOperator
        Operator <|-- TemporalOperator
        Operator <|-- Others
        Expression <|-- Operator

Thus, an `Expression` consists of multiple `Operators` which are either explicit, implicit, or temporal.
Consequently, addition, subtraction, and scaling with a field needs to be handled by the `Operator`.
