#!/usr/bin/env bash
set -ex

# uv run all files in pretrained_models/
for file in pretrained_models/*.py; do
    echo "Running $file"
    uv run "$file"
done
