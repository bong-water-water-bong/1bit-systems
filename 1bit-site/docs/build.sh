#!/usr/bin/env bash
# docs/build.sh — render docs/wiki/*.md into 1bit-site/docs/*.html
# Usage: bash 1bit-site/docs/build.sh [repo-root]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SITE="$(cd "$HERE/.." && pwd)"
ROOT="${1:-$(cd "$SITE/.." && pwd)}"
WIKI="$ROOT/docs/wiki"

if [ ! -d "$WIKI" ]; then
  echo "docs/build.sh: wiki source not found at $WIKI" >&2
  exit 1
fi

# Compile the tiny rustc-only renderer on demand.
BIN="$HERE/.build/render"
mkdir -p "$HERE/.build"
if [ ! -x "$BIN" ] || [ "$HERE/build.rs" -nt "$BIN" ]; then
  rustc -O "$HERE/build.rs" -o "$BIN"
fi

# Render every *.md to lowercase .html alongside.
count=0
for md in "$WIKI"/*.md; do
  name="$(basename "$md" .md)"
  lower="$(echo "$name" | tr '[:upper:]' '[:lower:]')"
  "$BIN" "$md" "$HERE/$lower.html" "$name"
  count=$((count + 1))
done

echo "docs/build.sh: rendered $count pages into $HERE/"
