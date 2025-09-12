#!/bin/bash
set -ex

uv run pytest --cov=torch_max_backend --cov-report=html --cov-report=term -n 7
