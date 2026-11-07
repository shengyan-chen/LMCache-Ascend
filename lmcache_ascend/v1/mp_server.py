# SPDX-License-Identifier: Apache-2.0
"""Explicit Ascend MP entry: install Host memory hooks, then run upstream CLI."""

# Standard
from collections.abc import Sequence
import sys


def main(argv: Sequence[str] | None = None) -> None:
    """Run the existing server command with Ascend's optional memory adaptation.

    Accept the same arguments as ``lmcache server``, without the ``server``
    subcommand. Normal imports complete before installing the hooks; no server
    module is executed a second time as __main__. Upstream owns CLI parsing,
    HTTP/MP startup, and shutdown.
    """
    # Third Party
    from lmcache.cli.main import main as lmcache_main

    # First Party
    from lmcache_ascend.v1.mp_host_memory import install_mp_host_memory

    install_mp_host_memory()
    lmcache_main(["server", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    main()
