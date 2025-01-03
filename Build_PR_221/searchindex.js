Search.setIndex({"docnames": ["basics/algorithms", "basics/containers", "basics/executor", "basics/fields", "basics/freeFunctions/fill", "basics/freeFunctions/map", "basics/freeFunctions/setField", "basics/index", "basics/macros", "basics/registerclass", "contributing", "datastructures/database", "datastructures/dictionary", "datastructures/index", "datastructures/unstructuredMesh", "dsl/equation", "dsl/index", "dsl/operator", "finiteVolume/cellCentred/boundaryConditions", "finiteVolume/cellCentred/case_study", "finiteVolume/cellCentred/index", "finiteVolume/cellCentred/operators", "finiteVolume/cellCentred/stencil", "index", "installation", "mpi_architecture", "timeIntegration/forwardEuler", "timeIntegration/index", "timeIntegration/rungeKutta"], "filenames": ["basics/algorithms.rst", "basics/containers.rst", "basics/executor.rst", "basics/fields.rst", "basics/freeFunctions/fill.rst", "basics/freeFunctions/map.rst", "basics/freeFunctions/setField.rst", "basics/index.rst", "basics/macros.rst", "basics/registerclass.rst", "contributing.rst", "datastructures/database.rst", "datastructures/dictionary.rst", "datastructures/index.rst", "datastructures/unstructuredMesh.rst", "dsl/equation.rst", "dsl/index.rst", "dsl/operator.rst", "finiteVolume/cellCentred/boundaryConditions.rst", "finiteVolume/cellCentred/case_study.rst", "finiteVolume/cellCentred/index.rst", "finiteVolume/cellCentred/operators.rst", "finiteVolume/cellCentred/stencil.rst", "index.rst", "installation.rst", "mpi_architecture.rst", "timeIntegration/forwardEuler.rst", "timeIntegration/index.rst", "timeIntegration/rungeKutta.rst"], "titles": ["Parallel Algorithms", "&lt;no title&gt;", "Executor", "Fields", "<code class=\"docutils literal notranslate\"><span class=\"pre\">fill</span></code>", "<code class=\"docutils literal notranslate\"><span class=\"pre\">map</span></code>", "<code class=\"docutils literal notranslate\"><span class=\"pre\">setField</span></code>", "Basics", "Macro Definitions", "Derived class discovery at compile time", "Contributing", "Database", "Dictionary", "Datastructures", "UnstructuredMesh", "Expression", "Domain Specific Language (DSL)", "Operator", "Boundary Conditions", "Case Study: The Gauss Green Div Kernel", "FiniteVolume", "Operators", "Stencil", "Welcome to NeoFOAM!", "Installation", "MPI Architecture", "Forward Euler", "Time Integration", "Runge Kutta"], "terms": {"ar": [0, 2, 3, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 23, 24, 25, 26, 27], "basic": [0, 3, 18, 23, 27], "build": [0, 8, 17, 23, 26, 27], "block": [0, 3, 24, 25], "implement": [0, 2, 3, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20, 25, 27], "advanc": [0, 26, 27], "kernel": [0, 3, 18, 20, 23], "To": [0, 2, 9, 10, 11, 12, 15, 16, 17, 18, 24, 25, 26], "simplifi": [0, 3, 9, 10, 11, 16], "we": [0, 2, 10, 19, 23, 24, 25], "provid": [0, 2, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 20, 24, 25, 26, 27, 28], "set": [0, 6, 8, 10, 11, 15, 18, 23, 24, 25, 26, 27, 28], "standard": [0, 8, 16, 24], "These": [0, 8], "can": [0, 2, 9, 10, 11, 12, 16, 17, 19, 20, 24, 25], "found": [0, 9, 10, 12, 17, 19, 20], "follow": [0, 3, 9, 10, 11, 12, 15, 16, 18, 19, 24, 25, 27], "file": [0, 8, 10, 19, 25], "includ": [0, 3, 10, 19, 20, 25], "neofoam": [0, 2, 3, 4, 5, 6, 8, 9, 11, 12, 15, 16, 17, 18, 19, 20, 24, 25, 26, 27, 28], "core": [0, 10, 15, 23, 25], "parallelalgorithm": 0, "hpp": [0, 3, 4, 5, 6, 7, 10, 19, 25], "test": [0, 10, 11, 12, 17, 19, 23, 24], "cpp": [0, 9, 10, 17, 19], "current": [0, 3, 14, 18, 20, 25, 27], "parallelfor": [0, 3, 19], "parallelreduc": 0, "The": [0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 20, 23, 24, 25, 26, 27, 28], "code": [0, 3, 8, 11, 12, 23, 24, 25], "show": [0, 3, 9, 11], "field": [0, 2, 4, 5, 6, 7, 11, 15, 17, 18, 19, 23, 26, 27, 28], "templat": [0, 3, 4, 5, 6, 9, 10, 11, 18, 25, 26, 28], "typenam": [0, 3, 4, 5, 6, 9, 11, 18, 25, 26, 28], "executor": [0, 3, 4, 5, 6, 7, 14, 17, 18, 19, 23, 25], "valuetyp": [0, 4, 6, 7, 10, 18, 25], "parallelforfieldkernel": 0, "void": [0, 2, 3, 4, 5, 6, 18, 25, 26, 28], "maybe_unus": 0, "const": [0, 2, 3, 4, 5, 6, 9, 10, 11, 12, 17, 18, 19, 25, 28], "exec": [0, 2, 4, 5, 6, 11, 17, 18], "auto": [0, 2, 4, 5, 6, 9, 11, 17, 18, 25, 26], "span": [0, 6, 17, 25], "constexpr": 0, "std": [0, 2, 3, 4, 5, 6, 9, 11, 12, 18, 25], "is_sam": 0, "remove_reference_t": 0, "serialexecutor": [0, 2, 4, 5, 6], "valu": [0, 3, 4, 6, 8, 10, 11, 12, 17, 18, 24, 25], "size_t": [0, 4, 5, 6, 11, 17, 18, 25], "i": [0, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 23, 24, 25, 26, 27], "0": [0, 4, 5, 6, 8, 11, 16, 17, 25], "size": [0, 3, 4, 5, 6, 10, 11, 25, 26], "els": 0, "us": [0, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 16, 17, 18, 19, 24, 25, 26, 28], "runon": 0, "kokko": [0, 2, 3, 18, 24, 25, 27, 28], "parallel_for": [0, 3, 18], "rangepolici": [0, 3, 18], "kokkos_lambda": [0, 3, 5, 17, 18], "base": [0, 9, 11, 18, 25, 27, 28], "type": [0, 2, 3, 10, 11, 12, 17, 18, 24, 25, 26, 27, 28], "function": [0, 3, 4, 5, 6, 9, 10, 11, 17, 18, 19, 25, 26], "either": [0, 2, 8, 15, 17, 24], "run": [0, 2, 10], "directli": [0, 11, 26, 27], "within": [0, 3, 15, 18, 25, 26], "loop": [0, 19, 25], "dispatch": [0, 3], "all": [0, 3, 9, 10, 11, 18, 19, 20, 25, 28], "non": [0, 10, 12, 25], "determin": [0, 8], "thu": [0, 3, 10, 14, 15, 24, 25], "gpu": [0, 2, 3, 10, 23, 25, 26], "gpuexecutor": [0, 2, 4, 5, 6, 18], "wa": 0, "addition": [0, 9, 24], "name": [0, 3, 9, 10, 11, 19, 25], "improv": 0, "visibl": 0, "profil": 0, "tool": [0, 9, 10, 24], "like": [0, 2, 3, 10, 11, 16, 23, 24], "nsy": 0, "final": [0, 10, 25], "assign": 0, "result": [0, 3, 16, 25], "given": [0, 4, 6, 8, 10, 16, 25], "here": [0, 10, 11, 19], "hold": [0, 3, 14, 15, 25], "data": [0, 3, 10, 11, 12, 14, 18, 25], "pointer": [0, 8], "devic": [0, 2], "defin": [0, 3, 4, 5, 6, 8, 9, 11, 12, 17, 25], "begin": [0, 25], "end": [0, 25], "sever": [0, 3, 10, 16, 24, 25, 28], "overload": 0, "exist": [0, 10, 11, 12, 23, 24, 25], "without": [0, 2, 12, 17], "an": [0, 2, 3, 8, 9, 10, 11, 12, 15, 16, 17, 18, 25], "explicitli": [0, 10, 19, 24], "rang": [0, 4, 5, 6, 18, 19], "learn": 0, "more": [0, 3, 11, 12, 16, 17, 25, 27], "how": [0, 9, 11, 15, 19, 25], "recommend": [0, 24], "check": [0, 2, 8, 9, 10, 12, 25], "correspond": [0, 3, 10, 19, 20], "unit": [0, 10, 12, 23, 25], "further": [0, 3, 9, 14, 17, 25], "detail": [0, 3, 9, 12, 14, 17, 24, 27], "free": [0, 3, 19, 26, 27], "fill": [0, 7], "map": [0, 6, 7, 9, 25], "setfield": [0, 7], "mpi": [2, 23], "x": [2, 8], "approach": [2, 9, 11, 16, 23, 25, 27], "parallel": [2, 3, 7, 23, 25, 28], "where": [2, 8, 9, 11, 25], "execut": [2, 10, 24, 26, 28], "space": [2, 16, 24, 25, 26, 28], "class": [2, 7, 10, 11, 12, 14, 15, 17, 18, 19, 23, 25, 26, 28], "interfac": [2, 11, 12, 17, 18, 27], "memori": [2, 3, 25, 28], "manag": [2, 9, 11, 25, 28], "specifi": [2, 8, 9, 18], "oper": [2, 3, 11, 15, 16, 18, 19, 23, 25, 26, 27, 28], "cpu": 2, "cpuexecutor": [2, 4, 5, 6], "openmp": [2, 25], "c": [2, 10, 16, 23], "thread": [2, 25], "combin": [2, 3, 16, 24, 25, 27, 28], "One": 2, "goal": [2, 23], "abil": 2, "easili": [2, 11, 24], "switch": [2, 24, 27], "between": [2, 11, 15, 24, 25, 27, 28], "differ": [2, 3, 15, 17, 24, 25], "time": [2, 7, 11, 15, 16, 23, 25, 26, 28], "e": [2, 10, 15, 23, 25], "enabl": [2, 8, 11, 24, 25], "have": [2, 9, 10, 11, 19, 24, 25], "recompil": 2, "thi": [2, 3, 8, 9, 10, 11, 12, 15, 16, 19, 23, 24, 25, 26, 27, 28], "achiev": [2, 17, 25], "pass": [2, 3, 9, 12, 16], "argument": [2, 9, 11, 19], "contain": [2, 3, 11, 14, 18, 25, 26], "shown": [2, 3, 9, 11, 17, 18, 24], "below": [2, 3, 9, 11, 17, 18, 24, 25], "gpuexec": 2, "cpuexec": 2, "scalar": [2, 3, 4, 5, 6, 11, 16, 17, 18, 25, 26, 27, 28], "gpufield": 2, "10": [2, 24, 25], "cpufield": 2, "variant": 2, "allow": [2, 3, 9, 11, 12, 16, 25, 27], "strategi": [2, 15], "alloc": [2, 3, 25], "runtim": [2, 9, 19, 27], "visit": [2, 18], "functor": 2, "struct": [2, 11], "cout": [2, 4, 5, 6, 25], "endl": [2, 4, 5, 6, 25], "pattern": [2, 11, 18, 19], "abov": [2, 11, 16, 18, 25], "would": [2, 25], "print": [2, 4, 5, 6, 8], "messag": [2, 8, 25], "depend": [2, 19, 24, 26, 27], "extend": [2, 11], "librari": [2, 8, 16, 23, 26, 27, 28], "addit": [2, 3, 9, 11, 15, 17, 25], "featur": [2, 10], "should": [2, 3, 8, 9, 10, 16, 24, 25], "two": [2, 3, 8, 9, 10, 25, 27], "same": [2, 16, 25], "equal": [2, 8], "api": [3, 18, 23], "probabl": [3, 18], "chang": [3, 10, 18, 19, 24], "futur": [3, 18, 23], "support": [3, 11, 14, 16, 17, 25, 27], "capabl": [3, 27, 28], "algebra": 3, "boundaryfield": [3, 18], "A": [3, 8, 10, 11, 16, 17, 18, 25], "friendli": 3, "datastructur": [3, 20, 23], "store": [3, 9, 10, 12, 14, 15, 18, 25], "boundari": [3, 14, 20, 23], "domainfield": [3, 18], "intern": [3, 19], "its": [3, 9, 11, 16, 25, 26], "besid": 3, "finit": [3, 18, 20], "volum": [3, 17, 18, 20], "geometricfieldmixin": 3, "mixin": 3, "mesh": [3, 10, 11, 14, 25], "volumefield": [3, 11, 17, 20, 26, 27, 28], "member": [3, 9, 10, 18, 19, 25], "notion": 3, "concret": 3, "condiditon": 3, "surfacefield": 3, "surfac": [3, 18], "equival": 3, "central": [3, 11], "element": [3, 5, 6, 25], "platform": [3, 23], "portabl": [3, 23], "cfd": 3, "framework": [3, 27], "perform": [3, 10, 11, 15, 16, 17, 25], "binari": [3, 23], "subtract": [3, 15], "multipl": [3, 15, 25], "In": [3, 10, 11, 15, 16, 18, 19, 25], "some": [3, 10, 25], "exampl": [3, 8, 9, 10, 11, 17, 25], "nodiscard": [3, 10], "t": [3, 5, 10, 11, 23, 26, 28], "rh": [3, 19], "exec_": 3, "size_": 3, "add": [3, 10, 11, 17], "return": [3, 5, 9, 10, 11, 17, 25], "creat": [3, 9, 10, 17, 24, 25], "temporari": [3, 16, 17], "mainli": [3, 11, 14], "call": [3, 19, 25], "stand": [3, 19], "which": [3, 10, 11, 15, 17, 18, 19, 23, 25], "fieldfreefunct": [3, 4, 5, 6], "turn": 3, "fieldbinaryop": 3, "actual": [3, 18, 19, 25], "lambda": [3, 11, 17], "our": [3, 25], "ultim": [3, 15, 18], "see": [3, 12, 19, 25], "document": [3, 9, 13, 23, 24], "type_identity_t": [3, 4, 6], "b": [3, 6, 8, 10], "va": 3, "vb": 3, "version": [3, 8, 19], "snippet": [3, 11], "highlight": 3, "anoth": [3, 25], "import": 3, "aspect": 3, "program": [3, 6, 8, 16, 25], "model": [3, 10], "also": [3, 9, 10, 12, 17, 19, 25], "deallocationg": 3, "finitevolum": [3, 10, 18, 19, 23], "cellcentr": [3, 10, 18, 19, 20], "folder": [3, 10], "namespac": [3, 20], "both": [3, 18, 24, 25, 26, 27], "deriv": [3, 7, 17, 18, 19, 23, 25], "from": [3, 9, 11, 14, 16, 17, 19, 23, 24, 25, 26], "handl": [3, 8, 15, 17, 25, 26, 28], "geometr": [3, 25], "inform": [3, 8, 10, 14, 15, 25, 27], "via": [3, 9, 19, 23, 27], "act": [3, 18, 27], "fundament": [3, 25], "structur": [3, 10, 12], "offer": [3, 8, 11, 27], "read": [3, 11, 14, 16], "write": 3, "internalfield": [3, 11, 18, 26], "vector": [3, 11, 15, 17, 25, 28], "condit": [3, 8, 20, 23], "correctboundarycondit": [3, 18, 26], "updat": [3, 10, 18, 24, 25], "": [3, 9, 10, 20, 24, 25, 27, 28], "compar": [3, 14, 26], "openfoam": [3, 9, 11, 14, 15, 16, 17, 18, 19], "volscalarfield": [3, 16], "volvectorfield": [3, 16], "voltensorfield": 3, "surfacescalarfield": 3, "surfacevectorfield": 3, "surfacetensorfield": 3, "respect": 3, "so": [3, 10, 14, 23], "branch": 3, "requir": [3, 10, 11, 16, 17, 24, 25, 26, 27], "when": [3, 8, 25, 26, 27], "iter": [3, 11, 18], "over": [3, 10, 18], "face": 3, "scalarfield": 3, "header": [4, 5, 6, 8, 10], "entir": [4, 6, 26], "subfield": [4, 5, 6], "pair": [4, 5, 6, 11], "specif": [4, 5, 6, 7, 9, 10, 11, 12, 17, 18, 20, 23, 24, 25, 28], "paramet": [4, 5, 6, 9, 10, 18], "If": [4, 5, 6, 10, 12, 25], "whole": [4, 5, 6], "ani": [4, 5, 6, 10, 11, 12, 25], "other": [4, 5, 6, 10, 15, 23, 25], "2": [4, 5, 6, 11, 16, 17, 25], "1": [4, 5, 6, 11, 16, 17, 25, 26], "copi": [4, 5, 6, 25], "host": [4, 5, 6], "hostfield": [4, 5, 6], "copytohost": [4, 5, 6], "appli": [5, 10, 18, 19, 25], "each": [5, 11, 25, 26, 27], "inner": 5, "fielda": 6, "fieldb": 6, "fieldc": 6, "note": [6, 24, 25, 27], "doe": [6, 25], "match": [6, 11], "exit": 6, "segfault": 6, "onli": [6, 8, 11, 16, 17, 18, 25], "last": 6, "overview": 7, "design": [7, 11, 16, 18], "cell": [7, 16, 18, 25], "centr": [7, 16], "algorithm": [7, 16, 20, 23, 25], "discoveri": [7, 23], "compil": [7, 10, 23, 24], "usag": [7, 8, 27], "macro": [7, 19, 23], "definit": [7, 17, 23], "info": 7, "error": [7, 11, 25], "debug": [8, 11, 24, 25], "report": 8, "assert": 8, "mechan": [8, 9], "log": 8, "across": [8, 25], "nf_debug": 8, "activ": 8, "cmake_build_typ": [8, 24], "nf_debug_info": 8, "relwithdebinfo": 8, "nf_debug_messag": 8, "nf_info": 8, "output": 8, "stream": 8, "nf_dinfo": 8, "nf_error_exit": 8, "abort": 8, "nf_throw": 8, "throw": [8, 12, 25], "neofoamexcept": 8, "nf_assert": 8, "fals": [8, 11, 12], "nf_assert_throw": 8, "nf_debug_assert": 8, "nf_debug_assert_throw": 8, "op": [8, 25], "releas": [8, 24, 25], "nf_assert_equ": 8, "thei": [8, 11, 19, 25], "nf_debug_assert_equ": 8, "nf_assert_equal_throw": 8, "nf_debug_assert_equal_throw": 8, "nf_ping": 8, "reach": 8, "certain": 8, "line": [8, 10, 25], "pure": [8, 10], "conveni": [8, 11], "stack": [8, 9, 10], "trace": 8, "gener": [8, 10, 12, 17, 25], "cpptrace": 8, "critic": 8, "occur": [8, 25], "ptr": [8, 9], "nullptr": [8, 28], "null": 8, "posit": [8, 25], "runtimeselectionfactori": 9, "regist": [9, 11, 19], "creation": 9, "factori": 9, "similar": [9, 11, 12, 17, 25], "select": [9, 14, 19], "static": [9, 11, 19], "initi": [9, 25], "explan": 9, "post": 9, "overflow": 9, "automat": [9, 24, 25, 26], "plugin": [9, 23], "architectur": [9, 11, 16, 23], "load": [9, 25], "unique_ptr": 9, "baseclass": 9, "derivedclass": 9, "registr": 9, "associ": 9, "string": [9, 11, 12, 25], "registerclass": [9, 19], "take": [9, 19], "list": [9, 10, 24], "insid": 9, "arg": 9, "public": [9, 11, 18], "registerdocument": 9, "kei": [9, 11, 12, 25, 27, 28], "keyexistsorerror": 9, "tabl": 9, "forward": [9, 23, 27, 28], "must": [9, 17, 18, 25], "doc": [9, 10, 11], "schema": 9, "after": [9, 10, 16, 24, 26], "been": [9, 10, 19], "instanti": [9, 17], "testderiv": 9, "baseclassdocument": 9, "retriev": [9, 17], "baseclassnam": 9, "around": [9, 28], "highli": [10, 23], "welcom": 10, "get": [10, 11, 12, 17], "you": [10, 24], "start": [10, 25], "review": 10, "adher": [10, 11], "howev": [10, 14, 16, 23, 24, 25], "most": [10, 11, 24, 25], "autom": 10, "For": [10, 11, 25, 26, 27], "format": [10, 11, 16], "relat": [10, 12], "enforc": 10, "clang": [10, 24], "tidi": 10, "configur": [10, 24, 26, 27, 28], "furthermor": 10, "adequ": 10, "licens": 10, "sourc": [10, 23, 26], "reus": 10, "linter": 10, "typo": 10, "obviou": 10, "spell": 10, "issu": [10, 16], "doesn": [10, 11], "stylist": 10, "rule": [10, 11], "rather": [10, 15, 16], "give": [10, 25], "advic": 10, "ambigu": 10, "situat": [10, 25], "mention": 10, "ration": 10, "decis": 10, "try": 10, "compli": 10, "guidelin": 10, "camelcas": 10, "capit": 10, "descript": [10, 24], "prefer": 10, "float": 10, "doubl": [10, 11, 25], "except": [10, 12], "getter": [10, 11], "indic": 10, "simpli": 10, "instead": [10, 18, 23], "expens": [10, 25], "comput": [10, 14, 16, 25, 26], "omit": 10, "abstract": 10, "aim": [10, 23, 25], "flat": 10, "inherit": [10, 11, 17, 26], "hierarchi": 10, "composit": 10, "avoid": [10, 25], "unintend": 10, "advis": 10, "refer": [10, 11, 12, 25], "might": [10, 23], "unstructur": 10, "outliv": 10, "object": [10, 11], "privat": [10, 11], "variabl": [10, 11], "suffix": 10, "_": 10, "order": [10, 19, 25, 26, 27, 28], "out": [10, 25], "g": [10, 15], "foo": 10, "inout": 10, "variat": 10, "u": [10, 11, 16], "locat": [10, 25], "consist": [10, 11, 15, 25], "src": [10, 19], "redund": 10, "ie": 10, "geometrymodel": 10, "finitevolumecellcentredgeometrymodel": 10, "want": [10, 23], "fix": [10, 18], "pleas": 10, "don": [10, 23], "hesit": 10, "open": [10, 24, 25], "pr": 10, "process": [10, 16, 25], "person": 10, "readi": [10, 23], "At": [10, 11], "least": 10, "one": [10, 11, 24, 25, 26, 28], "ideal": 10, "approv": 10, "befor": 10, "merg": 10, "make": [10, 11, 19], "sure": 10, "pipelin": 10, "succe": 10, "new": [10, 13, 17], "suffici": 10, "point": 10, "full": [10, 24, 25], "ci": 10, "hardwar": 10, "bug": 10, "entri": 10, "changelog": 10, "md": 10, "refactor": [10, 11], "skip": 10, "permiss": 10, "rebas": 10, "your": 10, "latest": [10, 24], "state": [10, 11], "main": [10, 15, 25], "yourself": 10, "author": 10, "small": 10, "medium": 10, "exce": 10, "1000": 10, "consid": 10, "break": 10, "up": [10, 25], "smaller": [10, 25, 26], "influenc": 10, "mean": [10, 23, 26], "discuss": [10, 19, 25], "signal": 10, "work": [10, 16, 23, 24], "ha": [10, 11, 16, 23, 25, 26], "finish": [10, 25], "whether": 10, "step": [10, 15, 16, 24, 25, 26], "command": [10, 24], "databas": [10, 13, 23], "cach": 10, "forc": 10, "rebuild": 10, "everi": 10, "push": 10, "aw": 10, "doxygen": [10, 24], "onlin": 10, "local": [10, 25], "do": [10, 14, 18, 25], "first": [10, 18, 25, 26], "sphinx": [10, 24], "instal": [10, 23], "system": [10, 15, 16, 24, 25], "second": 10, "cmake": [10, 23], "dneofoam_build_doc": 10, "ON": [10, 24], "target": 10, "html": 10, "docs_build": [10, 24], "built": 10, "directori": [10, 24], "view": 10, "index": [10, 11, 14, 18, 23, 25], "web": 10, "browser": 10, "firefox": 10, "altern": 10, "just": 10, "ad": [10, 13], "objectregistri": 11, "access": [11, 12, 16, 25], "modern": [11, 23], "softwar": [11, 23], "develop": [11, 16, 23, 24, 25], "best": 11, "practic": [11, 25], "emphas": 11, "modular": [11, 16], "testabl": 11, "loos": 11, "coupl": 11, "compon": [11, 28], "while": [11, 16, 25, 26], "simpl": [11, 16, 18, 25, 26], "inher": 11, "conflict": 11, "principl": 11, "encourag": 11, "relianc": 11, "codebas": 11, "less": 11, "flexibl": [11, 27], "adopt": [11, 16], "introduc": [11, 25], "potenti": 11, "bottleneck": 11, "challeng": 11, "contrast": [11, 16, 18], "valid": 11, "ensur": [11, 25], "integr": [11, 15, 16, 23, 26, 28], "predefin": 11, "prevent": [11, 24, 25], "inconsist": 11, "tightli": 11, "fvmesh": [11, 14], "easier": [11, 19], "diagram": 11, "illustr": 11, "relationship": 11, "n": [11, 26], "lowest": 11, "level": [11, 23], "dictionari": [11, 13, 23, 26, 27, 28], "id": 11, "python": [11, 12], "key1": [11, 12], "value1": 11, "key2": [11, 12], "3": [11, 15, 17, 25], "substr": 11, "4": 11, "doc_": 11, "possibl": [11, 15, 16, 18, 23, 25], "dict": [11, 12], "require_nothrow": 11, "As": [11, 18], "earlier": 11, "part": [11, 19, 25], "itself": [11, 23], "done": [11, 25], "fielddocu": 11, "keep": 11, "metadata": 11, "subcycl": 11, "fieldtyp": 11, "timeindex": 11, "iterationindex": 11, "int64_t": 11, "subcycleindex": 11, "validatefielddoc": 11, "appropri": 11, "setter": 11, "user": [11, 14, 17, 25], "fvcc": 11, "instanc": [11, 25], "db": 11, "newtestfieldcollect": 11, "registerfield": 11, "createfield": 11, "method": [11, 12, 14, 15, 16, 18, 23, 26, 27, 28], "expect": [11, 23], "createfunct": 11, "could": [11, 23, 25], "look": 11, "unstructuredmesh": [11, 13, 23], "volumeboundari": [11, 18, 20], "bc": 11, "patchi": 11, "insert": [11, 12], "fixedvalu": [11, 18], "push_back": 11, "ncell": [11, 17], "vf": 11, "find": 11, "resnam": 11, "fielddoc": 11, "constvolfield": 11, "somenam": 11, "somevalu": 11, "42": [11, 12], "boolean": 11, "filter": 11, "extens": [11, 23, 24], "through": [11, 25, 26, 27, 28], "eras": 11, "minim": [11, 16, 25, 26], "necessari": [11, 28], "domain": [11, 23], "customdocu": 11, "testvalu": 11, "validatecustomdoc": 11, "wrap": 11, "own": 11, "accessor": 11, "modifi": [11, 12, 18], "them": [11, 24], "collectionmixin": 11, "customcollect": 11, "bool": 11, "docs_": 11, "cc": 11, "emplac": 11, "true": [11, 12], "col": 11, "alreadi": 11, "boilerpl": 11, "focu": [11, 16, 25], "deliv": [12, 23], "complex": [12, 16, 25], "input": 12, "simul": [12, 16], "need": [12, 14, 15, 16, 17, 25], "It": [12, 14, 18, 24, 25, 26], "hello": 12, "int": [12, 19, 25], "well": 12, "43": 12, "out_of_rang": 12, "non_existent_kei": 12, "remov": [12, 24, 25], "sub": [12, 15], "group": 12, "togeth": [12, 25], "subdict": 12, "sdict": 12, "100": 12, "sdict2": 12, "boundarymesh": 13, "fieldcollect": 13, "queri": 13, "collect": 13, "relev": [14, 24], "repres": [14, 17, 18, 19], "grid": 14, "sinc": [14, 15, 19, 25], "construct": [14, 15, 25], "disc": 14, "convert": 14, "foamadapt": [14, 23], "continu": [14, 16, 25], "arrai": [14, 18], "patch": 14, "offset": 14, "unabl": 14, "help": [15, 16, 25], "formul": 15, "equat": [15, 16, 17, 26, 27, 28], "Its": [15, 25], "respons": [15, 18, 25], "li": 15, "answer": 15, "question": 15, "discret": [15, 19], "spatial": 15, "term": [15, 16, 17, 19, 25], "fvscheme": [15, 16], "solv": [15, 16, 25, 26, 27, 28], "fvsolut": [15, 16], "dsl": [15, 17, 19, 23], "evalu": [15, 16], "lazili": 15, "default": [15, 24, 25], "ti": 15, "delai": 15, "numer": [15, 16], "rk": [15, 16, 28], "even": 15, "lazi": [15, 16], "explicit": [15, 17, 18, 19, 26, 27, 28], "implicit": [15, 17, 18], "tempor": [15, 17], "consequ": [15, 17], "scale": [15, 17, 25], "concept": 16, "engin": [16, 25], "express": [16, 17, 23, 26, 28], "concis": 16, "readabl": 16, "form": [16, 25], "close": 16, "resembl": 16, "mathemat": 16, "represent": [16, 28], "littl": 16, "knowledg": 16, "scheme": 16, "physic": 16, "problem": [16, 25], "than": [16, 25], "reduc": [16, 17, 25], "effort": 16, "maintain": 16, "navier": 16, "stoke": 16, "fvvectormatrix": 16, "ueqn": 16, "fvm": 16, "ddt": 16, "div": [16, 20, 23], "phi": [16, 19], "laplacian": 16, "nu": 16, "fvc": 16, "grad": 16, "p": [16, 24], "piso": 16, "vectormatrix": 16, "diagon": 16, "off": [16, 24], "matrix": 16, "rau": 16, "hbya": 16, "constrainhbya": 16, "h": 16, "easi": 16, "understand": 16, "familiar": 16, "limit": 16, "due": 16, "solut": [16, 25, 26], "alwai": [16, 25], "spars": 16, "individu": 16, "eagerli": 16, "unnecessari": 16, "ldu": 16, "extern": [16, 26, 27], "linear": 16, "solver": [16, 25, 26], "discretis": 16, "tri": 16, "address": [16, 25], "better": 16, "optimis": 16, "number": [16, 17, 25], "coo": 16, "csr": 16, "pde": [16, 27], "sundial": [16, 27, 28], "bdf": 16, "heterogen": 16, "drop": 16, "replac": [16, 19, 23, 25], "imp": 16, "exp": [16, 28], "assembli": 16, "defer": 16, "till": 16, "henc": [16, 19], "major": [16, 25], "dure": 16, "That": 16, "assembl": 16, "scalabl": [17, 25], "coeffici": 17, "erasur": 17, "polymorph": 17, "divterm": 17, "diverg": [17, 19], "ddtterm": 17, "timeterm": 17, "fit": 17, "storag": 17, "abl": [17, 25], "scalingfield": 17, "sf": [17, 19], "customterm": 17, "constantscaledterm": 17, "constant": 17, "factor": 17, "fieldscaledterm": 17, "syntax": 17, "multiscaledterm": 17, "lambdascaledterm": 17, "operatormixin": 17, "virtual": [17, 18, 25], "explicitoper": [17, 26], "implicitoper": 17, "gettyp": [17, 25], "coeff": 17, "getcoeffici": 17, "draft": 18, "underli": 18, "noop": 18, "surfaceboundari": [18, 20], "attribut": 18, "boundarypatchmixin": 18, "center": 18, "volumetr": 18, "visitor": 18, "fvccscalarfixedvalueboundaryfield": 18, "bfield": 18, "fixedvaluebckernel": 18, "kernel_": 18, "mesh_": 18, "patchid_": 18, "start_": 18, "end_": 18, "uniformvalue_": 18, "logic": [18, 25, 28], "s_valu": 18, "s_refvalu": 18, "refvalu": 18, "uniformvalu": 18, "contigu": 18, "uniform": 18, "volfield": 18, "zerogradi": 18, "calcul": [18, 25], "section": [19, 25, 27], "explain": 19, "gaussgreendiv": 19, "nabla": 19, "cdot": [19, 26], "dv": 19, "particular": 19, "let": 19, "divoperatorfactori": 19, "computediv": 19, "correct": [19, 25], "common": [19, 24, 27], "commun": [19, 23], "written": 19, "d": [19, 24], "sum_f": 19, "s_f": 19, "phi_f": 19, "foral": 19, "owner": 19, "facei": 19, "gradtyp": 19, "sfssf": 19, "issf": 19, "iggrad": 19, "neighbour": 19, "bodi": 19, "threadsafe_add": 19, "threadsafe_sub": 19, "lh": 19, "case": [20, 23, 25], "studi": [20, 23], "gauss": [20, 23], "green": [20, 23], "project": [23, 24], "bring": 23, "By": [23, 24, 25], "reimplement": 23, "libfinitevolum": 23, "libopenfoam": 23, "compliant": 23, "20": 23, "high": [23, 28], "interoper": 23, "reason": [23, 25], "deviat": 23, "driven": 23, "contribut": 23, "everyon": 23, "preset": 23, "prerequisit": 23, "workflow": 23, "vscode": 23, "style": 23, "guid": 23, "collabor": 23, "pull": 23, "request": 23, "github": [23, 24], "label": 23, "languag": 23, "euler": [23, 27, 28], "rung": [23, 27], "kutta": [23, 27], "background": 23, "partit": 23, "abi": 23, "won": 23, "produc": 23, "serv": [23, 26, 27], "applic": [23, 27], "pimplefoam": 23, "against": 23, "demonstr": 23, "repositori": [23, 24], "modul": [23, 28], "search": 23, "page": 23, "clone": 24, "git": 24, "http": 24, "com": 24, "exasim": 24, "navig": 24, "cd": 24, "procedur": [24, 25], "mkdir": 24, "desiredbuildflag": 24, "chain": 24, "flag": [24, 25], "mode": 24, "neofoam_build_doc": 24, "neofoam_build_test": 24, "brows": 24, "option": [24, 26, 27], "ccmake": 24, "gui": 24, "prefix": 24, "neofoam_": 24, "kokkos_enable_cuda": 24, "kokkos_enable_hip": 24, "avail": [24, 25, 27], "commonli": 24, "product": 24, "ninja": 24, "chosen": 24, "wai": [24, 25], "bash": 24, "sudo": 24, "apt": 24, "pip": 24, "pre": 24, "commit": 24, "furo": 24, "breath": 24, "sitemap": 24, "ubuntu": 24, "24": 24, "04": 24, "16": 24, "gcc": 24, "libomp": 24, "dev": 24, "python3": 24, "essenti": [24, 25], "14": 24, "rm": 24, "usr": 24, "bin": 24, "ln": 24, "m": 24, "cpptool": 24, "button": 24, "tab": 24, "flask": 24, "icon": 24, "task": 24, "menu": 24, "ctrl": 24, "press": 24, "enter": 24, "larg": 25, "scientif": 25, "too": 25, "usabl": 25, "singl": 25, "broken": 25, "down": 25, "distribut": 25, "mani": 25, "share": 25, "rank": 25, "what": 25, "crucial": 25, "mask": 25, "cost": 25, "overhead": [25, 26], "broadli": 25, "conjunct": 25, "frequenc": 25, "brought": 25, "purpos": [25, 27], "seamlessli": 25, "suppli": 25, "typic": 25, "mpi_allreduc": 25, "reduceallscalar": 25, "reduceop": 25, "mpi_comm": 25, "comm": 25, "mpi_in_plac": 25, "reinterpret_cast": 25, "getop": 25, "reduct": 25, "environ": [25, 26, 27], "mpiinit": 25, "mpienviron": 25, "former": 25, "raii": [25, 28], "destructor": 25, "constructor": 25, "argc": 25, "char": 25, "argv": 25, "mpi_fin": 25, "onc": 25, "mpi_rank": 25, "mpi_siz": 25, "popul": 25, "mpi_comm_world": 25, "anywher": 25, "intend": 25, "split": 25, "longer": 25, "pars": 25, "With": 25, "place": 25, "mpi_commun": 25, "mpienv": 25, "sum": 25, "simplic": 25, "focus": 25, "reader": 25, "remind": 25, "terminologi": 25, "simplex": 25, "half": 25, "duplex": 25, "sender": 25, "receiv": 25, "vice": 25, "versa": 25, "direct": 25, "simultan": 25, "facilit": 25, "buffer": 25, "halfduplexcommbuff": 25, "send": 25, "pun": 25, "transfer": 25, "rel": 25, "never": 25, "laid": 25, "per": 25, "basi": 25, "therefor": 25, "kind": 25, "guard": 25, "rail": 25, "variou": 25, "until": 25, "fullduplexcommbuff": 25, "resourc": [25, 28], "wait": 25, "unload": 25, "de": 25, "unordered_map": 25, "comm_buff": 25, "sendsiz": 25, "receives": 25, "alldata": 25, "sendmap": 25, "assum": 25, "receivemap": 25, "obtain": 25, "initcomm": 25, "test_commun": 25, "commrank": 25, "sendbuff": 25, "getsendbuff": 25, "startcomm": 25, "waitcomplet": 25, "receivebuff": 25, "getreceivebuff": 25, "finalisecomm": 25, "later": 25, "inplac": 25, "remain": 25, "dead": 25, "lock": 25, "detect": 25, "hang": 25, "now": 25, "shift": 25, "overlap": 25, "present": 25, "dictat": 25, "stencil": 25, "miss": 25, "neighbor": 25, "halo": 25, "enough": 25, "nice": 25, "ranksimplexcommmap": 25, "arriv": 25, "role": 25, "pathwai": 25, "uniqu": 25, "identifi": 25, "worth": 25, "mai": [25, 26], "being": 25, "cours": 25, "made": 25, "sequenti": 25, "lead": 25, "divid": 25, "formal": 25, "world": 25, "dynam": 25, "balanc": 25, "metric": 25, "y_": 26, "y_n": 26, "delta": 26, "f": 26, "t_n": 26, "self": 26, "forwardeul": [26, 27], "timeintegratorbas": [26, 27], "straightforward": 26, "solutionfieldtyp": [26, 28], "eqn": 26, "sol": 26, "dt": [26, 28], "lightweight": [26, 27], "guarante": 26, "regardless": 26, "constraint": 26, "timedict": [26, 27, 28], "timeintegr": [26, 27, 28], "timestep": [26, 28], "solutionfield": [26, 28], "currenttim": [26, 28], "deltat": [26, 28], "fallback": [26, 27], "unavail": 26, "accuraci": 26, "higher": [26, 27], "synchron": 26, "partial": 27, "differenti": 27, "distinct": 27, "nativ": 27, "demand": 27, "robust": [27, 28], "seamless": 27, "wip": 27, "about": 27, "consider": 27, "leverag": 28, "rungekutta": 28, "wrapper": 28, "erkstep": 28, "convers": 28, "pdeexpr_": 28, "initsunerksolv": 28, "context": 28, "choos": 28}, "objects": {"": [[4, 0, 1, "_CPPv4I0EN7NeoFOAM4fillEvR5FieldI9ValueTypeEKNSt15type_identity_tI9ValueTypeEENSt4pairI6size_t6size_tEE", "NeoFOAM::fill"], [4, 1, 1, "_CPPv4I0EN7NeoFOAM4fillEvR5FieldI9ValueTypeEKNSt15type_identity_tI9ValueTypeEENSt4pairI6size_t6size_tEE", "NeoFOAM::fill::ValueType"], [4, 2, 1, "_CPPv4I0EN7NeoFOAM4fillEvR5FieldI9ValueTypeEKNSt15type_identity_tI9ValueTypeEENSt4pairI6size_t6size_tEE", "NeoFOAM::fill::a"], [4, 2, 1, "_CPPv4I0EN7NeoFOAM4fillEvR5FieldI9ValueTypeEKNSt15type_identity_tI9ValueTypeEENSt4pairI6size_t6size_tEE", "NeoFOAM::fill::range"], [4, 2, 1, "_CPPv4I0EN7NeoFOAM4fillEvR5FieldI9ValueTypeEKNSt15type_identity_tI9ValueTypeEENSt4pairI6size_t6size_tEE", "NeoFOAM::fill::value"], [18, 3, 1, "_CPPv4I0EN7NeoFOAM12finiteVolume11cellCentred15SurfaceBoundaryE", "NeoFOAM::finiteVolume::cellCentred::SurfaceBoundary"], [18, 1, 1, "_CPPv4I0EN7NeoFOAM12finiteVolume11cellCentred15SurfaceBoundaryE", "NeoFOAM::finiteVolume::cellCentred::SurfaceBoundary::ValueType"], [18, 3, 1, "_CPPv4I0EN7NeoFOAM12finiteVolume11cellCentred14VolumeBoundaryE", "NeoFOAM::finiteVolume::cellCentred::VolumeBoundary"], [18, 1, 1, "_CPPv4I0EN7NeoFOAM12finiteVolume11cellCentred14VolumeBoundaryE", "NeoFOAM::finiteVolume::cellCentred::VolumeBoundary::ValueType"], [5, 0, 1, "_CPPv4I00EN7NeoFOAM3mapEvR5FieldI1TEK5InnerNSt4pairI6size_t6size_tEE", "NeoFOAM::map"], [5, 1, 1, "_CPPv4I00EN7NeoFOAM3mapEvR5FieldI1TEK5InnerNSt4pairI6size_t6size_tEE", "NeoFOAM::map::Inner"], [5, 1, 1, "_CPPv4I00EN7NeoFOAM3mapEvR5FieldI1TEK5InnerNSt4pairI6size_t6size_tEE", "NeoFOAM::map::T"], [5, 2, 1, "_CPPv4I00EN7NeoFOAM3mapEvR5FieldI1TEK5InnerNSt4pairI6size_t6size_tEE", "NeoFOAM::map::a"], [5, 2, 1, "_CPPv4I00EN7NeoFOAM3mapEvR5FieldI1TEK5InnerNSt4pairI6size_t6size_tEE", "NeoFOAM::map::inner"], [5, 2, 1, "_CPPv4I00EN7NeoFOAM3mapEvR5FieldI1TEK5InnerNSt4pairI6size_t6size_tEE", "NeoFOAM::map::range"], [6, 0, 1, "_CPPv4I0EN7NeoFOAM8setFieldEvR5FieldI9ValueTypeEKNSt4spanIKNSt15type_identity_tI9ValueTypeEEEENSt4pairI6size_t6size_tEE", "NeoFOAM::setField"], [6, 1, 1, "_CPPv4I0EN7NeoFOAM8setFieldEvR5FieldI9ValueTypeEKNSt4spanIKNSt15type_identity_tI9ValueTypeEEEENSt4pairI6size_t6size_tEE", "NeoFOAM::setField::ValueType"], [6, 2, 1, "_CPPv4I0EN7NeoFOAM8setFieldEvR5FieldI9ValueTypeEKNSt4spanIKNSt15type_identity_tI9ValueTypeEEEENSt4pairI6size_t6size_tEE", "NeoFOAM::setField::a"], [6, 2, 1, "_CPPv4I0EN7NeoFOAM8setFieldEvR5FieldI9ValueTypeEKNSt4spanIKNSt15type_identity_tI9ValueTypeEEEENSt4pairI6size_t6size_tEE", "NeoFOAM::setField::b"], [6, 2, 1, "_CPPv4I0EN7NeoFOAM8setFieldEvR5FieldI9ValueTypeEKNSt4spanIKNSt15type_identity_tI9ValueTypeEEEENSt4pairI6size_t6size_tEE", "NeoFOAM::setField::range"]]}, "objtypes": {"0": "cpp:function", "1": "cpp:templateParam", "2": "cpp:functionParam", "3": "cpp:class"}, "objnames": {"0": ["cpp", "function", "C++ function"], "1": ["cpp", "templateParam", "C++ template parameter"], "2": ["cpp", "functionParam", "C++ function parameter"], "3": ["cpp", "class", "C++ class"]}, "titleterms": {"parallel": 0, "algorithm": 0, "executor": 2, "overview": [2, 3, 8], "design": 2, "field": [3, 25], "The": [3, 19], "valuetyp": 3, "class": [3, 9], "cell": 3, "centr": 3, "specif": [3, 16], "fill": 4, "descript": [4, 5, 6], "definit": [4, 5, 6, 8], "exampl": [4, 5, 6], "map": 5, "setfield": 6, "basic": 7, "macro": 8, "info": 8, "hpp": 8, "error": 8, "deriv": 9, "discoveri": 9, "compil": 9, "time": [9, 27], "usag": [9, 26, 28], "contribut": 10, "neofoam": [10, 23], "code": 10, "style": 10, "guid": 10, "collabor": 10, "via": 10, "pull": 10, "request": 10, "github": 10, "workflow": [10, 24], "label": 10, "build": [10, 24], "document": [10, 11], "databas": 11, "fieldcollect": 11, "queri": 11, "collect": 11, "ad": 11, "new": 11, "creat": 11, "custom": 11, "store": 11, "dictionari": 12, "datastructur": 13, "unstructuredmesh": 14, "boundarymesh": 14, "express": 15, "domain": 16, "languag": 16, "dsl": 16, "oper": [17, 21], "boundari": 18, "condit": 18, "volumefield": 18, "": 18, "case": 19, "studi": 19, "gauss": 19, "green": 19, "div": 19, "kernel": 19, "finitevolum": 20, "stencil": 22, "welcom": 23, "tabl": 23, "content": 23, "compat": 23, "openfoam": 23, "indic": 23, "instal": 24, "cmake": 24, "preset": 24, "prerequisit": 24, "vscode": 24, "mpi": 25, "architectur": 25, "background": 25, "commun": 25, "wrap": 25, "global": 25, "point": 25, "synchron": 25, "partit": 25, "futur": 25, "work": 25, "forward": 26, "euler": 26, "implement": [26, 28], "consider": 26, "integr": 27, "rung": 28, "kutta": 28}, "envversion": {"sphinx.domains.c": 3, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 9, "sphinx.domains.index": 1, "sphinx.domains.javascript": 3, "sphinx.domains.math": 2, "sphinx.domains.python": 4, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx.ext.todo": 2, "sphinx.ext.viewcode": 1, "sphinx": 60}, "alltitles": {"Parallel Algorithms": [[0, "parallel-algorithms"]], "Executor": [[2, "executor"]], "Overview": [[2, "overview"], [3, "overview"], [8, "overview"]], "Design": [[2, "design"]], "Fields": [[3, "fields"]], "The Field<ValueType> class": [[3, "the-field-valuetype-class"]], "Cell Centred Specific Fields": [[3, "cell-centred-specific-fields"]], "fill": [[4, "fill"]], "Description": [[4, "description"], [5, "description"], [6, "description"]], "Definition": [[4, "definition"], [5, "definition"], [6, "definition"]], "Example": [[4, "example"], [5, "example"], [6, "example"]], "map": [[5, "map"]], "setField": [[6, "setfield"]], "Basics": [[7, "basics"]], "Macro Definitions": [[8, "macro-definitions"]], "Info.hpp": [[8, "info-hpp"]], "Error.hpp": [[8, "error-hpp"]], "Derived class discovery at compile time": [[9, "derived-class-discovery-at-compile-time"]], "Usage": [[9, "usage"], [26, "usage"], [28, "usage"]], "Contributing": [[10, "contributing"]], "NeoFOAM Code Style Guide": [[10, "neofoam-code-style-guide"]], "Collaboration via Pull Requests": [[10, "collaboration-via-pull-requests"]], "Github Workflows and Labels": [[10, "github-workflows-and-labels"]], "Building the Documentation": [[10, "building-the-documentation"]], "Database": [[11, "database"]], "FieldCollection": [[11, "fieldcollection"]], "Query of document in a collection": [[11, "query-of-document-in-a-collection"]], "Adding a new collection and documents to the database": [[11, "adding-a-new-collection-and-documents-to-the-database"]], "Creating a Custom Document": [[11, "creating-a-custom-document"]], "Storing Custom Documents in a Custom Collection": [[11, "storing-custom-documents-in-a-custom-collection"]], "Dictionary": [[12, "dictionary"]], "Datastructures": [[13, "datastructures"]], "UnstructuredMesh": [[14, "unstructuredmesh"]], "BoundaryMesh": [[14, "boundarymesh"]], "Expression": [[15, "expression"]], "Domain Specific Language (DSL)": [[16, "domain-specific-language-dsl"]], "Operator": [[17, "operator"]], "Boundary Conditions": [[18, "boundary-conditions"]], "Boundary Conditions for VolumeField\u2019s": [[18, "boundary-conditions-for-volumefield-s"]], "Case Study: The Gauss Green Div Kernel": [[19, "case-study-the-gauss-green-div-kernel"]], "FiniteVolume": [[20, "finitevolume"]], "Operators": [[21, "operators"]], "Stencil": [[22, "stencil"]], "Welcome to NeoFOAM!": [[23, "welcome-to-neofoam"]], "Table of Contents": [[23, "table-of-contents"]], "Compatibility with OpenFOAM": [[23, "compatibility-with-openfoam"]], "Indices and tables": [[23, "indices-and-tables"]], "Installation": [[24, "installation"]], "Building with CMake Presets": [[24, "building-with-cmake-presets"]], "Prerequisites": [[24, "prerequisites"]], "Workflow with vscode": [[24, "workflow-with-vscode"]], "MPI Architecture": [[25, "mpi-architecture"]], "Background": [[25, "background"]], "Communication": [[25, "communication"]], "MPI Wrapping": [[25, "mpi-wrapping"]], "Global Communication": [[25, "global-communication"]], "Point-to-Point Communication": [[25, "point-to-point-communication"]], "Field Synchronization": [[25, "field-synchronization"]], "Partitioning": [[25, "partitioning"]], "Future Work": [[25, "future-work"]], "Forward Euler": [[26, "forward-euler"]], "Implementation": [[26, "implementation"], [28, "implementation"]], "Considerations": [[26, "considerations"]], "Time Integration": [[27, "time-integration"]], "Runge Kutta": [[28, "runge-kutta"]]}, "indexentries": {"neofoam::fill (c++ function)": [[4, "_CPPv4I0EN7NeoFOAM4fillEvR5FieldI9ValueTypeEKNSt15type_identity_tI9ValueTypeEENSt4pairI6size_t6size_tEE"]], "neofoam::map (c++ function)": [[5, "_CPPv4I00EN7NeoFOAM3mapEvR5FieldI1TEK5InnerNSt4pairI6size_t6size_tEE"]], "neofoam::setfield (c++ function)": [[6, "_CPPv4I0EN7NeoFOAM8setFieldEvR5FieldI9ValueTypeEKNSt4spanIKNSt15type_identity_tI9ValueTypeEEEENSt4pairI6size_t6size_tEE"]], "neofoam::finitevolume::cellcentred::surfaceboundary (c++ class)": [[18, "_CPPv4I0EN7NeoFOAM12finiteVolume11cellCentred15SurfaceBoundaryE"]], "neofoam::finitevolume::cellcentred::volumeboundary (c++ class)": [[18, "_CPPv4I0EN7NeoFOAM12finiteVolume11cellCentred14VolumeBoundaryE"]]}})