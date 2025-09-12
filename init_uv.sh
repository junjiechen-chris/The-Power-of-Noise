#!/bin/bash

# Prompt user to choose whether to save uv venv at local directory or network drive (workspace)
echo "Install UV virtual environment in $HOME/.uv-venvs/$(basename "$PWD") ?"
while true; do
    read -p "Install UV virtual environment in $HOME/.uv-venvs/$(basename "$PWD") ? " yn
    case $yn in
        [Yy]* ) export UV_PROJECT_ENVIRONMENT="$HOME/.uv-venvs/$(basename "$PWD")"; break;;
        [Nn]* ) export UV_PROJECT_ENVIRONMENT=""; break;;
        * ) echo "Please answer yes or no.";;
    esac
done

uv sync
uv run ipython kernel install --user --name=tpon
