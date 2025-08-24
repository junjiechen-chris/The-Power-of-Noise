#!/bin/bash

export UV_PROJECT_ENVIRONMENT="$HOME/.uv-venvs/$(basename "$PWD")"
uv sync