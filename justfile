fmt:
   cargo fmt

lint:
   cargo clippy

check:
   cargo check

test:
    cargo nextest run

run:
    cargo run --release