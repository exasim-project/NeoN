# SPDX-FileCopyrightText: 2023 - 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""``noIbm`` — the explicit opt-out.

No table, no band sweep: the bulk kernel alone. Bitwise equality with the plain
operator is *structural* here (there is nothing to apply), not a property
maintained by care — which is what makes it a usable baseline.
"""


class NoIbm:
    """Operator method that does nothing. Selected as ``solution["ibm"]``
    ``"noIbm"`` when a run wants the opt-out spelled out rather than implied by
    an absent key.

    ``requires_bodies = False`` is the whole implementation: ``evaluation``
    returns ``None`` for it, so no band is built, no row exists and no kernel
    launches.
    """

    name = "noIbm"
    kind = "operator"
    requires_bodies = False
