#!/usr/bin/env bash
# Regenerate studio-site/docs/ from docs/wiki/*.md.
# Compiles build.rs (std-only Rust) and runs it.
#
# No pandoc, no cargo deps — just rustc, which is already on any
# halo-ai dev box.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
src="${here}/../../docs/wiki"
out="${here}"

if [[ ! -d "$src" ]]; then
  echo "docs/wiki source missing: $src" >&2
  exit 1
fi

rustc -O "${here}/build.rs" -o "${here}/build_docs"
"${here}/build_docs" "$src" "$out"

# Optional: deploy to the local Caddy root.
if [[ "${1-}" == "--deploy" ]]; then
  sudo rsync -a --delete \
    --exclude build.rs --exclude build.sh --exclude build_docs \
    "${here}/" /srv/www/studio/docs/
  sudo chown -R caddy:caddy /srv/www/studio/docs/
fi
