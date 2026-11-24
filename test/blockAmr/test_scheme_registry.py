# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

import pytest
from blockamr.schemes.ddt_schemes import ForwardEuler, RungeKutta2, RungeKutta4
from blockamr.schemes.div_schemes import Linear, QUICK, Upwind, VanLeer
from blockamr.schemes.grad_schemes import CentralDiffGrad
from blockamr.schemes.laplacian_schemes import CentralDiffLaplacian
from blockamr.schemes.registry import lookup_scheme, resolve


@pytest.mark.parametrize(
    "operator, name, expected",
    [
        ("div", "upwind", Upwind),
        ("div", "linear", Linear),
        ("div", "vanLeer", VanLeer),
        ("div", "quick", QUICK),
        ("ddt", "Euler", ForwardEuler),
        ("ddt", "RK2", RungeKutta2),
        ("ddt", "RK4", RungeKutta4),
        ("laplacian", "central", CentralDiffLaplacian),
        ("grad", "central", CentralDiffGrad),
    ],
)
def test_every_name_resolves(operator, name, expected):
    assert resolve(operator, name) is expected


@pytest.mark.parametrize(
    "operator, name, expected",
    [
        ("div", "VANLEER", VanLeer),
        ("div", "Upwind", Upwind),
        ("ddt", "euler", ForwardEuler),
        ("ddt", "rk2", RungeKutta2),
        ("laplacian", "CENTRAL", CentralDiffLaplacian),
    ],
)
def test_resolve_is_case_insensitive(operator, name, expected):
    assert resolve(operator, name) is expected


def test_unknown_name_lists_valid_options():
    with pytest.raises(ValueError) as excinfo:
        resolve("div", "cubic")
    msg = str(excinfo.value)
    assert "cubic" in msg
    for option in ("upwind", "linear", "vanLeer", "quick"):
        assert option in msg

    with pytest.raises(ValueError, match="Euler"):
        resolve("ddt", "leapfrog")


def test_lookup_scheme_resolves_names_by_key():
    scheme = lookup_scheme({"div(phi,U)": "linear"}, ["div(phi,U)", "Div"], "div", Upwind())
    assert isinstance(scheme, Linear)


def test_lookup_scheme_default_fallback():
    """SchemesDict 'default' fallback semantics are preserved."""
    scheme = lookup_scheme({"default": "quick"}, ["div(phi,U)", "Div"], "div", Upwind())
    assert isinstance(scheme, QUICK)


def test_lookup_scheme_object_passthrough():
    obj = Linear()
    scheme = lookup_scheme({"Div": obj}, [None, "Div"], "div", Upwind())
    assert scheme is obj


def test_lookup_scheme_miss_returns_default():
    default = Upwind()
    assert lookup_scheme({}, ["div(phi,U)"], "div", default) is default
    assert lookup_scheme(None, ["div(phi,U)"], "div", default) is default
    assert lookup_scheme({"laplacian": "central"}, ["div(phi,U)", "Div"], "div", default) is default
