# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

class SchemesDict:
    """Runtime scheme selection dictionary."""

    def __init__(self, schemes=None):
        self._schemes = schemes or {}

    def lookup(self, name, default):
        if name in self._schemes:
            return self._schemes[name]
        if "default" in self._schemes:
            return self._schemes["default"]
        return default
