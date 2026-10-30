//! No tracked file may contain a NUL byte.
//!
//! Why this exists: a literal NUL in `bonefish-sys/src/lib.rs` (a `b"...\0"` written as the raw
//! byte instead of the escape) compiled fine, passed every check, and made git treat the whole
//! file as binary, which would have broken diffs and review of it. Nothing else would have
//! caught it.
//!
//! The file list comes from the git index (`git ls-files`), not from walking the disk, so other
//! worktrees and build output are never scanned. Every tracked file is expected to be text; a
//! tracked binary must be added to `ALLOWED_BINARY` on purpose, with a reason, which is the point.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Tracked files allowed to contain NUL bytes, each with the reason. Empty: everything tracked
/// today is text.
const ALLOWED_BINARY: &[(&str, &str)] = &[];

/// Byte offset of the first NUL in `path`, if any.
fn first_nul(path: &Path) -> std::io::Result<Option<usize>> {
    Ok(std::fs::read(path)?.iter().position(|b| *b == 0))
}

/// Every path in `rel_paths` (relative to `root`) that contains a NUL, with its offset.
fn files_with_nul(root: &Path, rel_paths: &[String]) -> Vec<(String, usize)> {
    rel_paths
        .iter()
        .filter_map(|rel| match first_nul(&root.join(rel)) {
            Ok(Some(at)) => Some((rel.clone(), at)),
            Ok(None) => None,
            Err(e) => panic!("cannot read tracked file {rel}: {e}"),
        })
        .collect()
}

/// The tracked files of the repository containing `root`, from the index.
fn tracked_files(root: &Path) -> Vec<String> {
    let out = Command::new("git")
        .args(["ls-files", "-z"])
        .current_dir(root)
        .output()
        .expect("git must be on PATH to enumerate tracked files");
    assert!(
        out.status.success(),
        "git ls-files failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
        .split(|b| *b == 0)
        .filter(|p| !p.is_empty())
        .map(|p| String::from_utf8(p.to_vec()).expect("tracked paths are UTF-8"))
        .collect()
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn no_tracked_file_contains_a_nul_byte() {
    let root = repo_root();
    let tracked = tracked_files(&root);
    // Positive control for the enumeration: the query sees files we know are tracked. A zero from
    // an enumeration that found nothing would prove nothing.
    for known in ["Cargo.toml", "src/lib.rs"] {
        assert!(
            tracked.iter().any(|t| t == known),
            "git ls-files did not list {known}; the enumeration is broken, not the repo clean"
        );
    }
    assert!(
        tracked.len() > 10,
        "suspiciously few tracked files: {}",
        tracked.len()
    );

    let found: Vec<(String, usize)> = files_with_nul(&root, &tracked)
        .into_iter()
        .filter(|(p, _)| !ALLOWED_BINARY.iter().any(|(a, _)| a == p))
        .collect();
    assert!(
        found.is_empty(),
        "tracked files contain NUL bytes (path, byte offset): {found:?}. A raw NUL in source is \
         almost always a mangled escape (write the two characters backslash-zero); if the file is \
         really binary, add it to ALLOWED_BINARY with a reason."
    );
}

/// Negative control: the scanner must flag a file that does contain a NUL, report where, and not
/// flag its clean neighbours. Without this, "no NULs found" could mean "the scanner cannot see
/// them".
#[test]
fn the_scanner_finds_a_planted_nul_and_only_that() {
    let dir = std::env::temp_dir().join(format!("bonefish-nul-check-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    std::fs::write(dir.join("clean.rs"), b"fn main() {}\n").unwrap();
    std::fs::write(dir.join("empty.txt"), b"").unwrap();
    std::fs::write(dir.join("bad.rs"), b"let m = b\"abc\0\";\n").unwrap();

    // The planted byte is a real NUL at offset 13, not the two-character escape in a string.
    let names = ["clean.rs", "empty.txt", "bad.rs"]
        .map(String::from)
        .to_vec();
    assert_eq!(
        files_with_nul(&dir, &names),
        vec![("bad.rs".to_string(), 13)]
    );

    // And the escape spelled as text (backslash, zero) is NOT a NUL: that is the fix, so it must
    // pass.
    std::fs::write(dir.join("escaped.rs"), br#"let m = b"abc\0";"#).unwrap();
    assert!(files_with_nul(&dir, &["escaped.rs".to_string()]).is_empty());
}

#[test]
fn allowed_binary_entries_have_reasons_and_exist() {
    let root = repo_root();
    let tracked = tracked_files(&root);
    for (path, reason) in ALLOWED_BINARY {
        assert!(
            !reason.trim().is_empty(),
            "{path}: an allowance needs a reason"
        );
        assert!(
            tracked.iter().any(|t| t == path),
            "{path} is allowed but not tracked: remove the stale allowance"
        );
    }
}
