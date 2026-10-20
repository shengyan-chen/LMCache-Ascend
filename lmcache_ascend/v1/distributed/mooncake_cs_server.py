# SPDX-License-Identifier: Apache-2.0
"""Explicit experimental entry point; accepts the usual ``lmcache server`` args."""

# Standard
import sys


def main() -> None:
    """Finish plugin initialization before installing MP hooks and entering CLI."""
    # First Party
    import lmcache  # noqa: F401
    from lmcache.cli.main import main as cli_main
    from lmcache_ascend.v1.distributed.mooncake_te_registration import (
        install,
        is_enabled,
    )

    if not is_enabled():
        raise RuntimeError("Set LMCACHE_ASCEND_MP_TE_UNMAP_HOST=1 for this launcher")
    install()
    sys.argv = ["lmcache", "server", *sys.argv[1:]]
    cli_main()


if __name__ == "__main__":
    main()
