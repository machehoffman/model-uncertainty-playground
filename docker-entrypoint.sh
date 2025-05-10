#!/bin/bash
set -e

# Install pre-commit hooks if not already installed
if [ ! -f .git/hooks/pre-commit ]; then
    echo "Installing pre-commit hooks..."
    pre-commit install
    pre-commit install --hook-type commit-msg
fi

# Execute the command passed to docker run
exec "$@" 