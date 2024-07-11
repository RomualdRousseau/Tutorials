set dotenv-load

# Default recipe to help
default: help

# Print the available recipes
help:
    @just --justfile {{justfile()}} --list

# Execute the pre-commit hooks
pre-commit:
    pre-commit run --files '**'
