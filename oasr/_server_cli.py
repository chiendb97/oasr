"""Console-script entry point for the Rust ``oasr-server`` front-end.

``pip install`` compiles the Rust serving core into the :mod:`oasr._core`
extension module via setuptools-rust (see ``pyproject.toml``'s
``[[tool.setuptools-rust.ext-modules]]``).  This thin wrapper forwards the
process ``argv`` into it so ``oasr-server ...`` on the ``PATH`` behaves like the
old standalone binary — same flags, same HTTP/gRPC servers, just shipped inside
the wheel instead of a separately-built executable.
"""

import sys


def main() -> None:
    try:
        from oasr._core import serve
    except ImportError as exc:  # pragma: no cover - surfaced only on broken installs
        raise SystemExit(
            "oasr._core is not available — the Rust serving extension was not built. "
            "Reinstall with a Rust toolchain present (e.g. `pip install -e .`)."
        ) from exc

    # Pass the full argv (including the program name at index 0); the Rust side
    # parses it with clap exactly as the standalone binary does.  `serve` blocks
    # until SIGINT/SIGTERM, then returns.
    serve(sys.argv)


if __name__ == "__main__":
    main()
