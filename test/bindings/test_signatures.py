# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import inspect
import re

import neon._neon as _neon

# nanobind shows the C++ type in a signature if it has no caster for it, e.g. when
# nanobind/stl/optional.h is not included. Such functions can't be used from Python.
UNCONVERTED_TYPE = re.compile(r"\b(?:std|NeoN)::\w+")
NB_FUNCTIONS = ("nb_func", "nb_method", "nb_bound_method")


def _bound_functions(obj, path, seen):
    """Yields the name and docstring of every function, method and property accessor."""
    if id(obj) in seen:
        return
    seen.add(id(obj))
    for name in dir(obj):
        try:
            attr = inspect.getattr_static(obj, name)
        except AttributeError:
            continue
        if type(attr).__name__ in NB_FUNCTIONS:
            yield f"{path}.{name}", attr.__doc__ or ""
        elif isinstance(attr, property):
            for accessor in (attr.fget, attr.fset):
                if type(accessor).__name__ in NB_FUNCTIONS:
                    yield f"{path}.{name}", accessor.__doc__ or ""
        elif inspect.ismodule(attr) and attr.__name__.startswith("neon"):
            yield from _bound_functions(attr, f"{path}.{name}", seen)
        elif inspect.isclass(attr) and attr.__module__.startswith("neon"):
            yield from _bound_functions(attr, f"{path}.{name}", seen)


def test_signatures_are_scanned():
    names = {name for name, _ in _bound_functions(_neon, "neon", set())}
    # a function, a method, a submodule function and a property
    assert {"neon.solve", "neon.ScalarVolumeField.exec", "neon.imp.laplacian", "neon.Vec3.x"} <= names


def test_all_signatures_use_python_types():
    unconverted = {
        name: sorted(set(UNCONVERTED_TYPE.findall(doc)))
        for name, doc in _bound_functions(_neon, "neon", set())
        if UNCONVERTED_TYPE.search(doc)
    }
    assert not unconverted, f"functions with C++ types in their signature: {unconverted}"
