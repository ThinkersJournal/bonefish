//! Per-test watchdog for the `bonefish-sys` tests (ROADMAP R17, bonefish#40: all dftr tests hung
//! about an hour in one full run and have not reproduced). `guarded` fails a test that overruns
//! its limit with a named panic. With `BONEFISH_TEST_WATCHDOG_SECS=<n>` set (the limit becomes
//! `n` seconds) it also prints a process-state dump to stderr first, so the next hang is
//! diagnosable. Without the variable there is no dump and the limit is the default.
//! A thread stack dump needs a debugger (`cdb` on Windows); without one the dump says so and
//! lists thread ids and wait states instead.
// Shared by several test binaries, each of which uses only part of it.
#![allow(dead_code)]
use std::time::Duration;

pub const FLAG: &str = "BONEFISH_TEST_WATCHDOG_SECS";
const DEFAULT_LIMIT: Duration = Duration::from_secs(120);

/// The configured limit and whether the diagnostic dump is on.
pub fn config() -> (Duration, bool) {
    match std::env::var(FLAG)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
    {
        Some(n) => (Duration::from_secs(n), true),
        None => (DEFAULT_LIMIT, false),
    }
}

/// Run a test body on its own thread and fail the test if it overruns the configured limit.
/// The stuck thread is left behind: the test process exits when the harness finishes.
pub fn guarded<F: FnOnce() + Send + 'static>(body: F) {
    let (limit, dump) = config();
    guarded_with(limit, dump, body);
}

/// `guarded` with limit and dump injected, so the timeout path can itself be tested.
pub fn guarded_with<F: FnOnce() + Send + 'static>(limit: Duration, dump: bool, body: F) {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body));
        let _ = tx.send(r);
    });
    match rx.recv_timeout(limit) {
        Ok(Ok(())) => {}
        Ok(Err(panic)) => std::panic::resume_unwind(panic),
        Err(_) => {
            if dump {
                eprintln!("{}", process_state());
            }
            panic!("test body did not finish within {limit:?} (see bonefish#40)")
        }
    }
}

fn run(cmd: &str, args: &[&str]) -> String {
    match std::process::Command::new(cmd).args(args).output() {
        Ok(o) => format!(
            "{}{}",
            String::from_utf8_lossy(&o.stdout),
            String::from_utf8_lossy(&o.stderr)
        ),
        Err(e) => format!("(could not run {cmd}: {e})\n"),
    }
}

/// Process state for a hang report: pid, threads with wait state, loaded modules (Windows),
/// children, and thread stacks when a debugger is on PATH.
pub fn process_state() -> String {
    let pid = std::process::id().to_string();
    let mut s = format!("=== bonefish test watchdog: process state, pid {pid} ===\n");
    if cfg!(windows) {
        let script = format!(
            "$p = Get-Process -Id {pid}; \"threads: $($p.Threads.Count)\"; \
             $p.Threads | ForEach-Object {{ \"tid $($_.Id) state $($_.ThreadState) wait $($_.WaitReason)\" }}; \
             \"modules:\"; $p.Modules | ForEach-Object {{ $_.FileName }}; \
             \"children:\"; Get-CimInstance Win32_Process -Filter \"ParentProcessId={pid}\" | \
             ForEach-Object {{ \"$($_.ProcessId) $($_.Name)\" }}"
        );
        s += &run("powershell", &["-NoProfile", "-Command", &script]);
        s += "stacks:\n";
        s += &run("cdb", &["-p", &pid, "-c", "~*k;qd"]);
    } else {
        s += "threads:\n";
        if let Ok(dir) = std::fs::read_dir("/proc/self/task") {
            for e in dir.flatten() {
                let p = e.path();
                let rd = |f: &str| std::fs::read_to_string(p.join(f)).unwrap_or_default();
                s += &format!(
                    "tid {:?} comm {} wchan {}\n",
                    e.file_name(),
                    rd("comm").trim(),
                    rd("wchan").trim()
                );
                s += &rd("stack");
            }
        }
        s += "children:\n";
        s += &run("pgrep", &["-l", "-P", &pid]);
    }
    s
}

/// Per-binary watchdog for tests that do not wrap their bodies: when the flag is set, the first
/// call starts a detached thread that, `n` seconds later, prints the process-state dump and
/// aborts the process (a hung binary is the bonefish#40 symptom; these binaries normally finish
/// in seconds). Idempotent; does nothing without the flag.
pub fn arm() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let (limit, on) = config();
        if on {
            std::thread::spawn(move || {
                std::thread::sleep(limit);
                // Raw stderr, not eprintln!: the harness captures eprintln! from test threads and
                // the abort below would discard it.
                use std::io::Write;
                let _ = writeln!(
                    std::io::stderr(),
                    "bonefish watchdog: test binary still running after {limit:?} (bonefish#40)
{}",
                    process_state()
                );
                std::process::abort();
            });
        }
    });
}
