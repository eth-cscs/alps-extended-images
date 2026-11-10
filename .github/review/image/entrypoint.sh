#!/usr/bin/env bash
set -euo pipefail
umask 077

# Configuration is baked into the read-only image; runtime state is private tmpfs.
# Disable upward and on-read project discovery, including PR-supplied AGENTS.md.
export OPENCODE_DISABLE_PROJECT_CONFIG=1
export XDG_CONFIG_HOME=/opt/reviewer/config
export HOME=/work/home
export XDG_DATA_HOME=/work/data
export XDG_CACHE_HOME=/work/cache
export XDG_STATE_HOME=/work/state
mkdir -p "$HOME" "$XDG_DATA_HOME" "$XDG_CACHE_HOME" "$XDG_STATE_HOME"
# The host checkout is owned by the runner, not the unprivileged container user.
git config --global --add safe.directory /input/pr-source
cd /work

# Both phases share private session storage; only the tools-disabled phase supplies review text.
exec node /opt/reviewer/run-review.mjs
