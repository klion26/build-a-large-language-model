format:
   pre-commit install
   pre-commit run fmt

lint:
   pre-commit install && pre-commit run clippy

check:
   pre-commit install && pre-commit run cargo-check