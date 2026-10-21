#!/usr/bin/env bash

set -e

if [[ -z "${VIRTUAL_ENV}" ]]; then
    >&2 echo "No active virtual environment found."
    exit 1
fi

pip install --upgrade pip
pip install --upgrade --force-reinstall -e .
pip install pytest==9.1.1 black==26.5.1

# Freeze the fully resolved environment. Poetry is not used to manage this venv,
# so this file -- not poetry.lock -- is what makes the environment reproducible.
pip freeze --exclude-editable > requirements.lock

echo
echo "Environment locked to requirements.lock ($(wc -l < requirements.lock) packages)."
