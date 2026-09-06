#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT
# SPDX-License-Identifier: Unlicense

"""Assert that an *unrepaired* wheel's shared libraries are relocatable.

``auditwheel repair`` and ``delocate-wheel`` rewrite RPATHs and install names
while repairing a wheel, so any check running after repair -- including the
``import neon`` smoke test in ``check_installed_wheel.py`` -- cannot tell
whether the build itself produced relocatable binaries. It only sees what the
repair tool patched up.

This script therefore runs from the cibuildwheel repair command, on the wheel
*before* it is repaired. It fails the build when a library would search for its
dependencies in the wheel-staging directory, which does not survive the
install. That is the failure mode bundled BLT-based dependencies (umpire, camp)
produce: BLT sets INSTALL_RPATH and, on macOS, CMAKE_INSTALL_NAME_DIR to
``${CMAKE_INSTALL_PREFIX}/lib``, and under scikit-build that prefix is a
temporary directory.

ELF and Mach-O are parsed directly rather than shelling out to readelf/otool so
that a missing binutils in a manylinux image cannot turn this check into a
silent pass.
"""

from __future__ import annotations

from pathlib import Path
import struct
import sys
import tempfile
import zipfile

# Absolute paths containing one of these are build- or staging-tree paths that
# do not exist on the machine that installs the wheel.
STAGING_MARKERS = (
    "_skbuild",
    "cibuildwheel",
    "/tmp/",
    "/var/folders/",
    "/project/",
    "/private/var/folders/",
)

# Loader-relative tokens, per binary format. These are not interchangeable: ld.so
# expands neither @loader_path nor @rpath, and dyld does not expand $ORIGIN, so a
# token borrowed from the other format is a dead RPATH entry rather than a
# relocatable one -- which is exactly the defect this check exists to catch.
ELF_RPATH_TOKENS = ("$ORIGIN", "${ORIGIN}")
MACHO_RPATH_TOKENS = ("@loader_path", "@executable_path")
# @rpath is resolved against the LC_RPATH list, so it is meaningful in an install
# name or a load command but never as an LC_RPATH entry itself.
MACHO_NAME_TOKENS = MACHO_RPATH_TOKENS + ("@rpath",)
ALL_LOADER_TOKENS = ELF_RPATH_TOKENS + MACHO_NAME_TOKENS

ELF, MACHO = "ELF", "Mach-O"

BINARY_SUFFIXES = (".so", ".dylib", ".pyd")


def rpath_tokens(fmt: str) -> tuple[str, ...]:
    """The loader-relative RPATH tokens the given format's loader can expand."""
    return ELF_RPATH_TOKENS if fmt == ELF else MACHO_RPATH_TOKENS


def is_loader_relative(path: str, fmt: str) -> bool:
    """True if this format's loader expands path relative to the loading binary."""
    return path.startswith(rpath_tokens(fmt))


def has_loader_token(path: str) -> bool:
    """True for any loader-relative token, whichever format it belongs to."""
    return path.startswith(ALL_LOADER_TOKENS)


def wrong_format_token(path: str, fmt: str) -> str | None:
    """Return the token in path that this format's loader cannot expand, if any."""
    if is_loader_relative(path, fmt):
        return None
    foreign = MACHO_NAME_TOKENS if fmt == ELF else ELF_RPATH_TOKENS
    return next((token for token in foreign if path.startswith(token)), None)


def is_staging(path: str) -> bool:
    """True for an absolute path that points into the build or staging tree."""
    if has_loader_token(path) or not path.startswith("/"):
        return False
    return any(marker in path for marker in STAGING_MARKERS)


def looks_binary(path: Path) -> bool:
    name = path.name
    return name.endswith(BINARY_SUFFIXES) or ".so." in name


# --------------------------------------------------------------------------
# ELF
# --------------------------------------------------------------------------

SHT_DYNAMIC = 6
DT_NULL, DT_NEEDED, DT_RPATH, DT_RUNPATH = 0, 1, 15, 29


def parse_elf(data: bytes) -> tuple[list[str], list[str]]:
    """Return (rpath entries, NEEDED names) from an ELF file's dynamic section."""
    is64 = data[4] == 2
    endian = "<" if data[5] == 1 else ">"

    if is64:
        (e_shoff,) = struct.unpack_from(endian + "Q", data, 0x28)
        e_shentsize, e_shnum = struct.unpack_from(endian + "HH", data, 0x3A)
        sh_fmt, sh_word = endian + "IIQQQQIIQQ", "Q"
    else:
        (e_shoff,) = struct.unpack_from(endian + "I", data, 0x20)
        e_shentsize, e_shnum = struct.unpack_from(endian + "HH", data, 0x2E)
        sh_fmt, sh_word = endian + "IIIIIIIIII", "I"

    dynamic = None
    sections = []
    for i in range(e_shnum):
        fields = struct.unpack_from(sh_fmt, data, e_shoff + i * e_shentsize)
        # sh_type, sh_offset, sh_size, sh_link
        sections.append((fields[1], fields[4], fields[5], fields[6]))
        if fields[1] == SHT_DYNAMIC:
            dynamic = sections[-1]

    if dynamic is None:
        return [], []

    _, dyn_off, dyn_size, dyn_link = dynamic
    _, str_off, str_size, _ = sections[dyn_link]
    strtab = data[str_off : str_off + str_size]

    def string_at(offset: int) -> str:
        end = strtab.find(b"\0", offset)
        return strtab[offset:end].decode("utf-8", "replace")

    rpaths: list[str] = []
    needed: list[str] = []
    entry_fmt = endian + sh_word * 2
    entry_size = struct.calcsize(entry_fmt)
    for pos in range(dyn_off, dyn_off + dyn_size, entry_size):
        d_tag, d_val = struct.unpack_from(entry_fmt, data, pos)
        if d_tag == DT_NULL:
            break
        if d_tag in (DT_RPATH, DT_RUNPATH):
            rpaths.extend(e for e in string_at(d_val).split(":") if e)
        elif d_tag == DT_NEEDED:
            needed.append(string_at(d_val))
    return rpaths, needed


# --------------------------------------------------------------------------
# Mach-O
# --------------------------------------------------------------------------

MH_MAGIC, MH_MAGIC_64 = 0xFEEDFACE, 0xFEEDFACF
FAT_MAGIC, FAT_MAGIC_64 = 0xCAFEBABE, 0xCAFEBABF
LC_LOAD_DYLIB, LC_ID_DYLIB = 0x0C, 0x0D
LC_LOAD_WEAK_DYLIB, LC_REEXPORT_DYLIB, LC_RPATH = 0x80000018, 0x8000001F, 0x8000001C
DYLIB_COMMANDS = (LC_LOAD_DYLIB, LC_ID_DYLIB, LC_LOAD_WEAK_DYLIB, LC_REEXPORT_DYLIB)


def _macho_slice(data: bytes, base: int) -> tuple[list[str], list[str]]:
    (magic,) = struct.unpack_from(">I", data, base)
    is64 = magic in (MH_MAGIC_64, 0xCFFAEDFE)
    endian = "<" if magic in (0xCFFAEDFE, 0xCEFAEDFE) else ">"
    (ncmds,) = struct.unpack_from(endian + "I", data, base + 16)
    pos = base + (32 if is64 else 28)

    rpaths: list[str] = []
    names: list[str] = []
    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack_from(endian + "II", data, pos)
        if cmdsize == 0:
            break
        if cmd == LC_RPATH or cmd in DYLIB_COMMANDS:
            (str_off,) = struct.unpack_from(endian + "I", data, pos + 8)
            raw = data[pos + str_off : pos + cmdsize]
            value = raw.split(b"\0", 1)[0].decode("utf-8", "replace")
            (rpaths if cmd == LC_RPATH else names).append(value)
        pos += cmdsize
    return rpaths, names


def parse_macho(data: bytes) -> tuple[list[str], list[str]]:
    """Return (LC_RPATH paths, dylib install/load names), merged over fat slices."""
    (magic,) = struct.unpack_from(">I", data, 0)
    if magic not in (FAT_MAGIC, FAT_MAGIC_64):
        return _macho_slice(data, 0)

    (nfat,) = struct.unpack_from(">I", data, 4)
    entry_size = 32 if magic == FAT_MAGIC_64 else 20
    rpaths: list[str] = []
    names: list[str] = []
    for i in range(nfat):
        base = 8 + i * entry_size
        if magic == FAT_MAGIC_64:
            (offset,) = struct.unpack_from(">Q", data, base + 8)
        else:
            (offset,) = struct.unpack_from(">I", data, base + 8)
        slice_rpaths, slice_names = _macho_slice(data, offset)
        rpaths.extend(slice_rpaths)
        names.extend(slice_names)
    return rpaths, names


def inspect(path: Path) -> tuple[str, list[str], list[str]] | None:
    """Return (format, rpaths, dependency names), or None if not ELF/Mach-O."""
    data = path.read_bytes()
    if len(data) < 8:
        return None
    if data[:4] == b"\x7fELF":
        return (ELF, *parse_elf(data))
    (magic,) = struct.unpack_from(">I", data, 0)
    if magic in (MH_MAGIC, MH_MAGIC_64, FAT_MAGIC, FAT_MAGIC_64, 0xCEFAEDFE, 0xCFFAEDFE):
        return (MACHO, *parse_macho(data))
    return None


# --------------------------------------------------------------------------


def check_tree(root: Path) -> list[str]:
    """Check every ELF/Mach-O library under root; return a list of problems."""
    problems: list[str] = []
    inspected: list[Path] = []
    umpire: tuple[str, list[str]] | None = None

    for path in sorted(root.rglob("*")):
        if not path.is_file() or not looks_binary(path):
            continue
        parsed = inspect(path)
        if parsed is None:
            continue
        fmt, rpaths, names = parsed
        inspected.append(path)
        rel = path.relative_to(root)

        if path.name.startswith("libumpire"):
            umpire = (fmt, rpaths)

        for entry in rpaths:
            if is_staging(entry):
                problems.append(f"{rel}: RPATH entry points into the staging tree: {entry}")
                continue
            foreign = wrong_format_token(entry, fmt)
            if foreign is not None:
                problems.append(
                    f"{rel}: RPATH entry uses {foreign}, which the {fmt} loader "
                    f"does not expand: {entry}"
                )
        for name in names:
            if is_staging(name):
                problems.append(f"{rel}: dependency recorded by absolute staging path: {name}")

    # A glob that matches nothing must not read as a clean bill of health.
    if not inspected:
        problems.append(f"no ELF or Mach-O libraries found under {root} -- nothing was checked")
    elif not any(p.name.startswith("_neon") for p in inspected):
        problems.append("the _neon extension module was not found in the wheel")

    # Regression guard: umpire's dependency on camp is the case that motivated
    # this check, and BLT's own RPATH is absolute. The token must be one this
    # binary's own loader expands -- $ORIGIN on a dylib is as useless as no RPATH.
    if umpire is not None:
        fmt, umpire_rpaths = umpire
        if umpire_rpaths and not any(is_loader_relative(e, fmt) for e in umpire_rpaths):
            problems.append(
                f"libumpire has no {fmt} loader-relative RPATH entry "
                f"(expected one of {rpath_tokens(fmt)}): {umpire_rpaths}"
            )

    print(f"checked {len(inspected)} shared libraries in {root.name}")
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {Path(argv[0]).name} <wheel>", file=sys.stderr)
        return 2

    wheel = Path(argv[1])
    if not wheel.is_file():
        print(f"error: no such wheel: {wheel}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(root)
        problems = check_tree(root)

    if problems:
        print(f"\n{wheel.name} is not relocatable:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print(f"{wheel.name}: all libraries are relocatable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
