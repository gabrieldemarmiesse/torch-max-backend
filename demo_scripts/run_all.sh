#!/usr/bin/env bash
set -ex

# Run every demo from the directory containing this script.
script_dir="${BASH_SOURCE[0]%/*}"
for file in "$script_dir"/*.py; do
    echo "Running $file"
    uv run "$file"
done
