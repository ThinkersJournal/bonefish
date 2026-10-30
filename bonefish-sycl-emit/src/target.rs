//! Proposed `sycl:` target-capability vocabulary (KISS-Classify §6.8), as code.
//!
//! **A proposal, not a registered namespace**: KISS records the vocabulary, and the maintainer's
//! annex lives in KISS. This module is the maintainer's side of KISS-CLASSIFY-6.8-0014: every
//! token carries a derivability witness, here the entry in DPC++'s own architecture table
//! (`sycl/ext/oneapi/experimental/device_architecture.def`), and `tests/sycl_target_tokens.rs`
//! is the named gate that evaluates it.
//!
//! # Grammar
//!
//! ```text
//! sycl:spir64                  portable: JIT-compiled SPIR-V (`-fsycl-targets=spir64`)
//! sycl:intel-gpu-<arch>        ahead-of-time for one Intel GPU architecture
//! ```
//!
//! `<arch>` is the canonical name of an `intel_gpu_*` entry in the table with every `_` replaced
//! by `-`, so the token is the `-fsycl-targets` value spelled with hyphens. **Aliases never get a
//! token**: `intel_gpu_dg2_g10` is another name for `intel_gpu_acm_g10`, the same cell, and a
//! second token would split it under the byte-exact match rule (§6.8-0002).

/// One `__SYCL_ARCHITECTURE` / `__SYCL_ARCHITECTURE_ALIAS` line of the table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArchEntry {
    /// The enumerator name, e.g. `intel_gpu_acm_g10`.
    pub name: String,
    /// `true` for `__SYCL_ARCHITECTURE_ALIAS`: another name for an existing architecture.
    pub alias: bool,
}

/// Why a name has no token.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TokenError {
    /// Not in the table at all.
    Unknown(String),
    /// An alias of another entry; use the canonical name.
    Alias(String),
    /// Present and canonical, but not an Intel GPU (outside this namespace's scope).
    NotIntelGpu(String),
}

/// The portable token: JIT-compiled SPIR-V, the default `icpx -fsycl` target.
pub const JIT_TOKEN: &str = "sycl:spir64";

/// Parse the architecture table. Lines that are not an entry (comments, `_CONDITIONAL`, blanks)
/// are skipped; a trailing `//` comment is ignored.
#[must_use]
pub fn parse_architecture_def(text: &str) -> Vec<ArchEntry> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.split("//").next().unwrap_or("").trim();
        let (alias, rest) = if let Some(r) = line.strip_prefix("__SYCL_ARCHITECTURE_ALIAS(") {
            (true, r)
        } else if let Some(r) = line.strip_prefix("__SYCL_ARCHITECTURE(") {
            (false, r)
        } else {
            continue;
        };
        if let Some(name) = rest.split(',').next() {
            let name = name.trim();
            if !name.is_empty() {
                out.push(ArchEntry {
                    name: name.to_string(),
                    alias,
                });
            }
        }
    }
    out
}

/// The `sycl:` token for an architecture name, or the reason it has none.
///
/// # Errors
///
/// [`TokenError`] for an unknown name, an alias, or a non-Intel-GPU entry.
pub fn token_for(name: &str, table: &[ArchEntry]) -> Result<String, TokenError> {
    let Some(entry) = table.iter().find(|e| e.name == name) else {
        return Err(TokenError::Unknown(name.to_string()));
    };
    if entry.alias {
        return Err(TokenError::Alias(name.to_string()));
    }
    if !name.starts_with("intel_gpu_") {
        return Err(TokenError::NotIntelGpu(name.to_string()));
    }
    Ok(format!("sycl:{}", name.replace('_', "-")))
}

/// Every `sycl:intel-gpu-*` token the table yields, in table order.
#[must_use]
pub fn derive_tokens(table: &[ArchEntry]) -> Vec<String> {
    table
        .iter()
        .filter_map(|e| token_for(&e.name, table).ok())
        .collect()
}

/// The tokens in `tokens` that no entry of `table` produces: the KISS-CLASSIFY-6.8-0014
/// direction (*every capability NAMED is REACHABLE*). [`JIT_TOKEN`] is witnessed by the compiler
/// accepting `-fsycl-targets=spir64`, not by the table, so it is skipped here.
#[must_use]
pub fn unwitnessed(tokens: &[&str], table: &[ArchEntry]) -> Vec<String> {
    let derived = derive_tokens(table);
    tokens
        .iter()
        .filter(|t| **t != JIT_TOKEN && !derived.iter().any(|d| d == **t))
        .map(|t| (*t).to_string())
        .collect()
}

/// The proposed vocabulary: each token with the first DPC++ release whose table (or, for the
/// portable token, whose driver) produces it. The vocabulary only grows (a new architecture is
/// an additive spelling, as for `cuda:`), so a token is never removed because an older compiler
/// lacks it; the gate checks a token's witness only against compilers at or after its release.
/// `intel_gpu_cri` is absent from the 2026.1 table and present in 2026.2's (observed in CI).
/// The gate checks this list against the installed table in both directions.
pub const TOKENS: &[(&str, &str)] = &[
    ("sycl:spir64", "2026.1"),
    ("sycl:intel-gpu-bdw", "2026.1"),
    ("sycl:intel-gpu-skl", "2026.1"),
    ("sycl:intel-gpu-kbl", "2026.1"),
    ("sycl:intel-gpu-cfl", "2026.1"),
    ("sycl:intel-gpu-apl", "2026.1"),
    ("sycl:intel-gpu-glk", "2026.1"),
    ("sycl:intel-gpu-whl", "2026.1"),
    ("sycl:intel-gpu-aml", "2026.1"),
    ("sycl:intel-gpu-cml", "2026.1"),
    ("sycl:intel-gpu-icllp", "2026.1"),
    ("sycl:intel-gpu-ehl", "2026.1"),
    ("sycl:intel-gpu-tgllp", "2026.1"),
    ("sycl:intel-gpu-rkl", "2026.1"),
    ("sycl:intel-gpu-adl-s", "2026.1"),
    ("sycl:intel-gpu-adl-p", "2026.1"),
    ("sycl:intel-gpu-adl-n", "2026.1"),
    ("sycl:intel-gpu-dg1", "2026.1"),
    ("sycl:intel-gpu-acm-g10", "2026.1"),
    ("sycl:intel-gpu-acm-g11", "2026.1"),
    ("sycl:intel-gpu-acm-g12", "2026.1"),
    ("sycl:intel-gpu-pvc", "2026.1"),
    ("sycl:intel-gpu-pvc-vg", "2026.1"),
    ("sycl:intel-gpu-mtl-u", "2026.1"),
    ("sycl:intel-gpu-mtl-h", "2026.1"),
    ("sycl:intel-gpu-arl-h", "2026.1"),
    ("sycl:intel-gpu-bmg-g21", "2026.1"),
    ("sycl:intel-gpu-bmg-g31", "2026.1"),
    ("sycl:intel-gpu-lnl-m", "2026.1"),
    ("sycl:intel-gpu-ptl-h", "2026.1"),
    ("sycl:intel-gpu-ptl-u", "2026.1"),
    ("sycl:intel-gpu-wcl", "2026.1"),
    ("sycl:intel-gpu-nvl-s", "2026.1"),
    ("sycl:intel-gpu-nvl-u", "2026.1"),
    ("sycl:intel-gpu-nvl-p", "2026.1"),
    ("sycl:intel-gpu-cri", "2026.2"),
];

/// The token strings, in order.
#[must_use]
pub fn token_names() -> Vec<&'static str> {
    TOKENS.iter().map(|(t, _)| *t).collect()
}

/// The tokens whose witness an installed DPC++ `installed = (year, minor)` should satisfy.
#[must_use]
pub fn tokens_since(installed: (u32, u32)) -> Vec<&'static str> {
    TOKENS
        .iter()
        .filter(|(_, since)| parse_release(since).is_some_and(|r| r <= installed))
        .map(|(t, _)| *t)
        .collect()
}

/// `"2026.1"` or `"2026.1.0"` -> `(2026, 1)`.
#[must_use]
pub fn parse_release(s: &str) -> Option<(u32, u32)> {
    let mut it = s.split('.');
    Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
__SYCL_ARCHITECTURE(unknown, 0x9900000000000000)
__SYCL_ARCHITECTURE(intel_gpu_acm_g10, 0x000000030dc00800)       // Alchemist G10
__SYCL_ARCHITECTURE_ALIAS(intel_gpu_dg2_g10, intel_gpu_acm_g10)  // Alchemist G10
__SYCL_ARCHITECTURE(intel_gpu_pvc, 0x000000030f000700)
__SYCL_ARCHITECTURE(nvidia_gpu_sm_90, 0x0100000000009000)
// __SYCL_ARCHITECTURE(intel_gpu_commented_out, 1)
";

    #[test]
    fn parses_entries_and_aliases_and_skips_comments() {
        let t = parse_architecture_def(SAMPLE);
        let names: Vec<&str> = t.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "unknown",
                "intel_gpu_acm_g10",
                "intel_gpu_dg2_g10",
                "intel_gpu_pvc",
                "nvidia_gpu_sm_90"
            ]
        );
        assert!(t[2].alias && !t[1].alias);
    }

    #[test]
    fn canonical_names_get_tokens() {
        let t = parse_architecture_def(SAMPLE);
        assert_eq!(
            token_for("intel_gpu_acm_g10", &t).unwrap(),
            "sycl:intel-gpu-acm-g10"
        );
        assert_eq!(
            derive_tokens(&t),
            ["sycl:intel-gpu-acm-g10", "sycl:intel-gpu-pvc"]
        );
    }

    #[test]
    fn an_alias_has_no_token_of_its_own() {
        let t = parse_architecture_def(SAMPLE);
        // Positive control: its canonical name does.
        assert!(token_for("intel_gpu_acm_g10", &t).is_ok());
        assert_eq!(
            token_for("intel_gpu_dg2_g10", &t),
            Err(TokenError::Alias("intel_gpu_dg2_g10".into()))
        );
    }

    #[test]
    fn unknown_and_non_intel_names_are_refused() {
        let t = parse_architecture_def(SAMPLE);
        assert_eq!(
            token_for("intel_gpu_nope", &t),
            Err(TokenError::Unknown("intel_gpu_nope".into()))
        );
        assert_eq!(
            token_for("nvidia_gpu_sm_90", &t),
            Err(TokenError::NotIntelGpu("nvidia_gpu_sm_90".into()))
        );
    }

    #[test]
    fn tokens_obey_the_kiss_charset() {
        // KISS-Classify §6.8-0005: no `|`, `;`, `/`, whitespace or control byte; and this
        // vocabulary is lowercase [a-z0-9-] after the single `:`.
        for tok in token_names() {
            let (ns, rest) = tok.split_once(':').expect("one colon");
            assert_eq!(ns, "sycl");
            assert!(!rest.contains(':') && !rest.is_empty(), "{tok}");
            assert!(
                rest.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-'),
                "{tok}"
            );
        }
    }

    #[test]
    fn tokens_are_unique() {
        let mut seen = std::collections::BTreeSet::new();
        for tok in token_names() {
            assert!(seen.insert(tok), "duplicate {tok}");
        }
    }

    #[test]
    fn release_filtering_is_monotonic_and_parses_versions() {
        assert_eq!(parse_release("2026.1"), Some((2026, 1)));
        assert_eq!(parse_release("2026.2.0"), Some((2026, 2)));
        assert_eq!(parse_release("nope"), None);
        let old = tokens_since((2026, 1));
        let new = tokens_since((2026, 2));
        assert!(old.iter().all(|t| new.contains(t)));
        assert_eq!(new.len(), old.len() + 1);
        assert!(new.contains(&"sycl:intel-gpu-cri") && !old.contains(&"sycl:intel-gpu-cri"));
        assert!(tokens_since((2025, 0)).is_empty());
    }
}
