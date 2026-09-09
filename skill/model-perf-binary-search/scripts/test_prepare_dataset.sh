#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_ROOT="$(mktemp -d)"
trap 'rm -rf "$TEST_ROOT"' EXIT

SHARED_DATA="$TEST_ROOT/shared/data"
mkdir -p "$SHARED_DATA"
printf '%s\n' '{"id":"alpha"}' > "$SHARED_DATA/alpha.jsonl"

eval "$(bash "$SCRIPT_DIR/prepare_dataset.sh" "$SHARED_DATA" "$SHARED_DATA/alpha.jsonl")"
test "$DATASET" = "$SHARED_DATA/alpha.jsonl"
test ! -e "$TEST_ROOT/work"

if bash "$SCRIPT_DIR/prepare_dataset.sh" "$SHARED_DATA" "alpha.jsonl" >/dev/null 2>&1; then
  echo "relative dataset unexpectedly accepted" >&2
  exit 1
fi

if bash "$SCRIPT_DIR/prepare_dataset.sh" "$SHARED_DATA" "$SHARED_DATA/missing.jsonl" >/dev/null 2>&1; then
  echo "missing dataset unexpectedly accepted" >&2
  exit 1
fi

: > "$SHARED_DATA/empty.jsonl"
if bash "$SCRIPT_DIR/prepare_dataset.sh" "$SHARED_DATA" "$SHARED_DATA/empty.jsonl" >/dev/null 2>&1; then
  echo "empty dataset unexpectedly accepted" >&2
  exit 1
fi

printf '%s\n' '{"id":"outside"}' > "$TEST_ROOT/outside.jsonl"
if bash "$SCRIPT_DIR/prepare_dataset.sh" "$SHARED_DATA" "$TEST_ROOT/outside.jsonl" >/dev/null 2>&1; then
  echo "outside dataset unexpectedly accepted" >&2
  exit 1
fi

if bash "$SCRIPT_DIR/prepare_dataset.sh" "$SHARED_DATA" "$SHARED_DATA" >/dev/null 2>&1; then
  echo "dataset directory unexpectedly accepted" >&2
  exit 1
fi

ln -s "$TEST_ROOT/outside.jsonl" "$SHARED_DATA/outside-alias.jsonl"
if bash "$SCRIPT_DIR/prepare_dataset.sh" "$SHARED_DATA" "$SHARED_DATA/outside-alias.jsonl" >/dev/null 2>&1; then
  echo "escaping dataset symlink unexpectedly accepted" >&2
  exit 1
fi

printf '%s\n' '{"id":"hidden"}' > "$SHARED_DATA/.hidden.jsonl"
eval "$(bash "$SCRIPT_DIR/prepare_dataset.sh" "$SHARED_DATA" "$SHARED_DATA/.hidden.jsonl")"
test "$DATASET" = "$SHARED_DATA/.hidden.jsonl"

ln -s "$SHARED_DATA/alpha.jsonl" "$SHARED_DATA/alpha-alias.jsonl"
eval "$(bash "$SCRIPT_DIR/prepare_dataset.sh" "$SHARED_DATA" "$SHARED_DATA/alpha-alias.jsonl")"
test "$DATASET" = "$SHARED_DATA/alpha.jsonl"

printf 'PASS\n'
