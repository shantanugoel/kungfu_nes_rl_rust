#!/usr/bin/env bash
set -euo pipefail
root="$(cd "${1:-.}" && pwd)"
output="/tmp/kungfu_rl_sources.txt"

rg --files "$root" \
  -g 'src/**' \
  -g 'scripts/**' \
  -g 'Cargo.toml' \
  -g 'README.md' \
  -g 'RAM_MAP.md' \
  -g 'rust_toolchain.toml' \
  | LC_ALL=C sort > /tmp/kungfu_rl_sources.list

: > "$output"
while IFS= read -r file; do
  if [[ -f "$file" ]]; then
    rel="${file#$root/}"
    printf '===== %s =====\n' "$rel" >> "$output"
    cat "$file" >> "$output"
    printf '\n\n' >> "$output"
  fi
done < /tmp/kungfu_rl_sources.list

if command -v pbcopy >/dev/null 2>&1; then
  cat "$output" | pbcopy
  echo "Copied $(wc -l < "$output") lines to clipboard from $output"
else
  echo "pbcopy not found. Output at $output"
fi
