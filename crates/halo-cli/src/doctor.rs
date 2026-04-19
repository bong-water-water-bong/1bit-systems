// `halo doctor` — health check across the stack. Exit 0 green, 1 warn, 2 fail.

use anyhow::Result;
use std::process::Command;
use std::time::Duration;

use crate::status::SERVICES;

#[derive(Copy, Clone)]
enum Outcome { Ok, Warn, Fail }

impl Outcome {
    fn glyph(self) -> &'static str { match self { Self::Ok => "●", Self::Warn => "◉", Self::Fail => "○" } }
    fn tag(self)   -> &'static str { match self { Self::Ok => "ok",  Self::Warn => "warn", Self::Fail => "fail" } }
}

fn row(name: &str, out: Outcome, detail: &str) {
    println!("  {}  {:<20} {:<5} {}", out.glyph(), name, out.tag(), detail);
}

fn cmd_ok(bin: &str, args: &[&str]) -> bool {
    Command::new(bin).args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status().map(|s| s.success()).unwrap_or(false)
}

fn cmd_out(bin: &str, args: &[&str]) -> Option<String> {
    Command::new(bin).args(args).output().ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
}

fn check_gpu() -> (Outcome, String) {
    match cmd_out("/opt/rocm/bin/rocminfo", &[]) {
        Some(s) if s.contains("gfx1151") => (Outcome::Ok, "gfx1151 present".into()),
        Some(_)                          => (Outcome::Fail, "rocminfo ran but no gfx1151".into()),
        None                             => (Outcome::Fail, "rocminfo not found at /opt/rocm/bin".into()),
    }
}

fn check_kernel() -> (Outcome, String) {
    match cmd_out("uname", &["-r"]) {
        Some(s) => {
            let v = s.trim();
            let major = v.split('.').next().and_then(|n| n.parse::<u32>().ok()).unwrap_or(0);
            if major >= 7 { (Outcome::Ok, format!("kernel {v}")) }
            else           { (Outcome::Warn, format!("kernel {v} — NPU driver wants 7.x")) }
        }
        None => (Outcome::Fail, "uname failed".into()),
    }
}

fn check_service(unit: &str, port: u16) -> (Outcome, String) {
    let active    = cmd_ok("systemctl", &["--user", "is-active", "--quiet", unit]);
    let listening = port == 0 || cmd_out("ss", &["-lnt"])
        .map(|s| s.lines().any(|l| l.contains(&format!("127.0.0.1:{port}"))))
        .unwrap_or(false);
    match (active, listening) {
        (true,  true)  => (Outcome::Ok,   "active + listening".into()),
        (true,  false) => (Outcome::Warn, "active, no port".into()),
        (false, _)     => (Outcome::Fail, "inactive".into()),
    }
}

async fn check_http(url: &str) -> (Outcome, String) {
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .danger_accept_invalid_certs(true)
        .build() { Ok(c) => c, Err(e) => return (Outcome::Fail, e.to_string()) };
    match client.get(url).send().await {
        Ok(r) if r.status().is_success() => (Outcome::Ok,   format!("{} {}", r.status().as_u16(), url)),
        Ok(r)                            => (Outcome::Warn, format!("{} {}", r.status().as_u16(), url)),
        Err(e)                           => (Outcome::Fail, format!("{url}: {e}")),
    }
}

fn check_pi() -> (Outcome, String) {
    if cmd_ok("ping", &["-c", "1", "-W", "1", "100.64.0.4"]) {
        (Outcome::Ok, "100.64.0.4 reachable".into())
    } else {
        (Outcome::Warn, "pi not reachable (archive offline?)".into())
    }
}

pub async fn run() -> Result<()> {
    let mut warn = 0u32;
    let mut fail = 0u32;
    let mut tally = |o: Outcome| match o {
        Outcome::Warn => warn += 1,
        Outcome::Fail => fail += 1,
        Outcome::Ok   => {}
    };

    println!("─── host ────────────────────────────────────");
    let (o, d) = check_gpu();    tally(o); row("gpu",    o, &d);
    let (o, d) = check_kernel(); tally(o); row("kernel", o, &d);

    println!("\n─── services ────────────────────────────────");
    for (short, unit, port) in SERVICES {
        let (o, d) = check_service(unit, *port);
        tally(o);
        row(short, o, &d);
    }

    println!("\n─── endpoints ───────────────────────────────");
    for (name, url) in &[
        ("v1 models", "http://127.0.0.1:8080/v1/models"),
        ("v2 models", "http://127.0.0.1:8180/v1/models"),
    ] {
        let (o, d) = check_http(url).await;
        tally(o);
        row(name, o, &d);
    }

    println!("\n─── network ─────────────────────────────────");
    let (o, d) = check_pi(); tally(o); row("pi-archive", o, &d);

    println!("\n{} warn, {} fail", warn, fail);
    std::process::exit(if fail > 0 { 2 } else if warn > 0 { 1 } else { 0 });
}
