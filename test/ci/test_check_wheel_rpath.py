#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT
# SPDX-License-Identifier: Unlicense

"""Tests for ci/check_wheel_rpath.py.

The wheel check is a CI gate, so it must fail on a broken wheel rather than
pass vacuously. The end-to-end cases compile real shared libraries with the
platform compiler and give them the exact RPATH/install-name shape that BLT
produces, so the parsers are exercised against genuine ELF/Mach-O rather than
hand-written fixtures.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ci"))

import check_wheel_rpath as cwr  # noqa: E402


IS_MACOS = sys.platform == "darwin"
EXT = ".dylib" if IS_MACOS else ".so"
CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
needs_cc = pytest.mark.skipif(CC is None, reason="no C compiler to build fixture libraries")


def build_library(
    directory: Path,
    name: str,
    rpaths: list[str],
    install_name: str | None = None,
    new_dtags: bool = True,
) -> Path:
    """Compile an empty shared library carrying the given RPATH entries."""
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / f"{name}.c"
    source.write_text("int neon_probe(void) { return 0; }\n")
    library = directory / f"lib{name}{EXT}"

    cmd = [CC, "-shared", "-fPIC", str(source), "-o", str(library)]
    for rpath in rpaths:
        cmd += ["-Wl,-rpath," + rpath]
    if install_name and IS_MACOS:
        cmd += ["-Wl,-install_name," + install_name]
    if not IS_MACOS and not new_dtags:
        # ld emits DT_RUNPATH by default; this produces the older DT_RPATH instead.
        cmd += ["-Wl,--disable-new-dtags"]
    subprocess.run(cmd, check=True, capture_output=True)
    source.unlink()
    return library


def make_wheel(tmp_path: Path, libraries: list[Path], with_extension: bool = True) -> Path:
    """Pack libraries into a wheel-shaped zip, mirroring the real layout."""
    wheel = tmp_path / "neon-0.0.0-cp312-cp312-test.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for library in libraries:
            archive.write(library, f"lib/{library.name}")
        if with_extension:
            suffix = sysconfig.get_config_var("EXT_SUFFIX") or EXT
            archive.write(libraries[0], f"neon/_neon{suffix}")
        archive.writestr("neon/__init__.py", "")
    return wheel


@pytest.mark.parametrize(
    "path, expected",
    [
        ("$ORIGIN", False),
        ("$ORIGIN/../lib", False),
        ("@loader_path/../lib", False),
        ("@rpath/libcamp.dylib", False),
        ("/usr/local/cuda/lib64", False),
        ("/usr/lib", False),
        ("libcamp.so", False),
        ("/tmp/tmp1a2b3c/wheel/platlib/lib", True),
        ("/private/var/folders/x9/T/tmpabcd/lib", True),
        ("/project/_skbuild/lib", True),
        ("/home/runner/work/NeoN/NeoN/_skbuild/lib", True),
    ],
)
def test_is_staging(path: str, expected: bool) -> None:
    assert cwr.is_staging(path) is expected


def test_empty_tree_is_not_a_pass(tmp_path: Path) -> None:
    """A glob matching nothing must be reported, not silently accepted."""
    problems = cwr.check_tree(tmp_path)
    assert any("nothing was checked" in problem for problem in problems)


def test_missing_extension_module_is_reported(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cwr, "inspect", lambda path: ([], []))
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / f"libNeoN{EXT}").write_bytes(b"\x7fELF" + b"\0" * 64)
    problems = cwr.check_tree(tmp_path)
    assert any("_neon extension module was not found" in problem for problem in problems)


@needs_cc
def test_relocatable_wheel_passes(tmp_path: Path) -> None:
    loader = "@loader_path" if IS_MACOS else "$ORIGIN"
    libraries = [
        build_library(tmp_path / "build", "umpire", [loader, f"{loader}/../lib"], "@rpath/libumpire.dylib"),
        build_library(tmp_path / "build", "camp", [loader], "@rpath/libcamp.dylib"),
    ]
    assert cwr.main(["check", str(make_wheel(tmp_path, libraries))]) == 0


@needs_cc
def test_staging_rpath_is_rejected(tmp_path: Path) -> None:
    """The BLT failure mode: INSTALL_RPATH left at the staging prefix."""
    staging = "/tmp/tmp0k2j/wheel/platlib/lib"
    libraries = [build_library(tmp_path / "build", "umpire", [staging], "@rpath/libumpire.dylib")]
    assert cwr.main(["check", str(make_wheel(tmp_path, libraries))]) == 1


@needs_cc
@pytest.mark.skipif(not IS_MACOS, reason="install names are a Mach-O concept")
def test_staging_install_name_is_rejected(tmp_path: Path) -> None:
    """The macOS half: CMAKE_INSTALL_NAME_DIR left at the staging prefix."""
    staging = "/tmp/tmp0k2j/wheel/platlib/lib"
    libraries = [
        build_library(tmp_path / "build", "umpire", ["@loader_path"], f"{staging}/libumpire.dylib")
    ]
    assert cwr.main(["check", str(make_wheel(tmp_path, libraries))]) == 1


@needs_cc
def test_umpire_without_loader_relative_rpath_is_rejected(tmp_path: Path) -> None:
    libraries = [build_library(tmp_path / "build", "umpire", ["/usr/local/lib"], "@rpath/libumpire.dylib")]
    assert cwr.main(["check", str(make_wheel(tmp_path, libraries))]) == 1


@needs_cc
@pytest.mark.skipif(IS_MACOS, reason="DT_RPATH is an ELF concept")
def test_legacy_dt_rpath_is_read(tmp_path: Path) -> None:
    """Libraries built with the older DT_RPATH tag must be inspected too."""
    library = build_library(
        tmp_path / "build", "umpire", ["/tmp/tmp0k2j/wheel/platlib/lib"], new_dtags=False
    )
    rpaths, _ = cwr.inspect(library)
    assert rpaths == ["/tmp/tmp0k2j/wheel/platlib/lib"]
    assert cwr.main(["check", str(make_wheel(tmp_path, [library]))]) == 1
