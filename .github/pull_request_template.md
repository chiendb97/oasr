## What and why

<!-- What does this change, and what problem does it solve? Link the issue: "Fixes #123". -->

## How it was validated

<!--
Which GPU (and compute capability), which checkpoints, which dtype, offline or
streaming. Paste the `external assets:` table pytest prints at the end of the
run. A `MISSING` asset means the tests covering it were skipped.
-->

## Checklist

- [ ] Lint is clean: `black`, `isort`, `ruff check` (and `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings` from `rust/` for Rust changes)
- [ ] Tests were added or updated, and run with `set -a; source .env; set +a`
- [ ] A new variant extends a registry rather than the engine core (see `docs/architecture.md`)
- [ ] Docs in `docs/` are updated where behaviour changed, with no benchmark numbers in `README.md`, `AGENTS.md` or `docs/`
