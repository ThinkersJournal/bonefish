// SPDX-License-Identifier: MIT OR Apache-2.0
//! `bonefish-bench`: see the library docs. Prints one JSON object per line on stdout; notes and
//! errors go to stderr.

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = bonefish_bench::cli(&args) {
        eprintln!("bonefish-bench: {e}");
        std::process::exit(1);
    }
}
