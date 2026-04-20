// docs/build.rs — tiny rustc-only markdown → HTML renderer.
// Compile: rustc -O build.rs -o .build/render
// Run:     ./.build/render <input.md> <output.html> <page-title>
//
// Not a full CommonMark implementation. Handles the subset our wiki
// actually uses: ATX headings (#..####), paragraphs, fenced code
// blocks (```), inline code (`x`), bold (**x**), italic (*x*),
// links [t](u), unordered/ordered lists, blockquotes, horizontal
// rules, GitHub-style pipe tables. Escapes HTML in all text.
//
// Output: a full HTML page with topbar + left sidebar + right TOC.

use std::env;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: {} <in.md> <out.html> <slug>", args[0]);
        return ExitCode::from(2);
    }
    let input = match fs::read_to_string(&args[1]) {
        Ok(s) => s,
        Err(e) => { eprintln!("read {}: {}", args[1], e); return ExitCode::from(2); }
    };
    let slug = &args[3];
    let (body_html, headings) = render_markdown(&input);
    let (title, page_title) = extract_title(&headings, slug);
    // Strip the first <h1> from body — wrap_page re-emits it above the
    // docs-body column, and showing it twice looks silly.
    let body_html = strip_first_h1(&body_html);
    let html = wrap_page(&body_html, &headings, &title, &page_title, slug);
    if let Some(parent) = Path::new(&args[2]).parent() {
        let _ = fs::create_dir_all(parent);
    }
    let mut f = match fs::File::create(&args[2]) {
        Ok(f) => f,
        Err(e) => { eprintln!("write {}: {}", args[2], e); return ExitCode::from(2); }
    };
    if let Err(e) = f.write_all(html.as_bytes()) {
        eprintln!("write: {}", e);
        return ExitCode::from(2);
    }
    ExitCode::SUCCESS
}

struct Heading { level: u8, text: String, id: String }

fn strip_first_h1(body: &str) -> String {
    // Find "<h1" ... "</h1>\n" (only the first one, and only if it's the
    // very first element in the body).
    let trimmed = body.trim_start();
    if !trimmed.starts_with("<h1") { return body.to_string(); }
    match trimmed.find("</h1>") {
        Some(end) => {
            let after = &trimmed[end + "</h1>".len()..];
            after.trim_start_matches('\n').to_string()
        }
        None => body.to_string(),
    }
}

fn extract_title(h: &[Heading], slug: &str) -> (String, String) {
    for head in h {
        if head.level == 1 { return (head.text.clone(), head.text.clone()); }
    }
    let pretty = slug.replace('-', " ");
    (pretty.clone(), pretty)
}

fn slugify(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_dash = false;
    for c in s.chars() {
        let c = c.to_ascii_lowercase();
        if c.is_ascii_alphanumeric() {
            out.push(c); prev_dash = false;
        } else if !prev_dash && !out.is_empty() {
            out.push('-'); prev_dash = true;
        }
    }
    while out.ends_with('-') { out.pop(); }
    if out.is_empty() { "section".to_string() } else { out }
}

fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => o.push_str("&amp;"),
            '<' => o.push_str("&lt;"),
            '>' => o.push_str("&gt;"),
            '"' => o.push_str("&quot;"),
            _   => o.push(c),
        }
    }
    o
}

// Inline render: **bold**, *em*, `code`, [t](u). Order: code first
// (shields everything inside), then links, then bold, then italic.
// All slicing is by byte indices into a UTF-8 &str (which is safe on
// char boundaries because we only split at ASCII markers like `, [, *).
fn render_inline(s: &str) -> String {
    // pass 1: extract code spans; replace with placeholders
    let mut code_spans: Vec<String> = Vec::new();
    let mut buf = String::new();
    let mut rest = s;
    loop {
        match rest.find('`') {
            None => { buf.push_str(rest); break; }
            Some(start) => {
                buf.push_str(&rest[..start]);
                let after = &rest[start+1..];
                match after.find('`') {
                    None => { buf.push('`'); rest = after; }
                    Some(end) => {
                        let inner = &after[..end];
                        let idx = code_spans.len();
                        code_spans.push(format!("<code>{}</code>", esc(inner)));
                        buf.push_str(&format!("\u{0001}C{}\u{0002}", idx));
                        rest = &after[end+1..];
                    }
                }
            }
        }
    }
    let mut out = esc(&buf);

    // links [text](url)
    out = render_links(&out);
    // bold then italic
    out = render_emph(&out, "**", "strong");
    out = render_emph(&out, "*", "em");

    // restore code spans
    for (idx, span) in code_spans.iter().enumerate() {
        let tok = format!("\u{0001}C{}\u{0002}", idx);
        out = out.replace(&tok, span);
    }
    out
}

fn render_links(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    loop {
        match rest.find('[') {
            None => { out.push_str(rest); return out; }
            Some(lb) => {
                out.push_str(&rest[..lb]);
                let after = &rest[lb+1..];
                let close = match after.find(']') {
                    None => { out.push('['); rest = after; continue; }
                    Some(c) => c,
                };
                let tail = &after[close+1..];
                if !tail.starts_with('(') {
                    out.push('['); rest = after; continue;
                }
                let rp = match tail[1..].find(')') {
                    None => { out.push('['); rest = after; continue; }
                    Some(r) => r,
                };
                let label = &after[..close];
                let url = &tail[1..1+rp];
                let href = rewrite_href(url);
                out.push_str(&format!("<a href=\"{}\">{}</a>", href, label));
                rest = &tail[1+rp+1..];
            }
        }
    }
}

fn rewrite_href(url: &str) -> String {
    // drop ./ prefix; lowercase *.md → *.html
    let mut u = url.trim_start_matches("./").to_string();
    if let Some(stripped) = u.strip_suffix(".md") {
        u = format!("{}.html", stripped.to_ascii_lowercase());
    } else if let Some(idx) = u.find(".md#") {
        let (a, b) = u.split_at(idx);
        u = format!("{}.html{}", a.to_ascii_lowercase(), &b[3..]);
    }
    u
}

fn render_emph(s: &str, marker: &str, tag: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    loop {
        match rest.find(marker) {
            None => { out.push_str(rest); return out; }
            Some(start) => {
                out.push_str(&rest[..start]);
                let after = &rest[start+marker.len()..];
                match after.find(marker) {
                    None => { out.push_str(&rest[start..]); return out; }
                    Some(end) => {
                        let inner = &after[..end];
                        if inner.is_empty() || inner.starts_with(' ') {
                            out.push_str(&rest[start..start+marker.len()]);
                            rest = &rest[start+marker.len()..];
                        } else {
                            out.push_str(&format!("<{t}>{i}</{t}>", t=tag, i=inner));
                            rest = &after[end+marker.len()..];
                        }
                    }
                }
            }
        }
    }
}

fn render_markdown(md: &str) -> (String, Vec<Heading>) {
    let lines: Vec<&str> = md.lines().collect();
    let mut out = String::new();
    let mut headings: Vec<Heading> = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        // fenced code
        if line.trim_start().starts_with("```") {
            let lang = line.trim_start().trim_start_matches("```").trim().to_string();
            let mut body = String::new();
            i += 1;
            while i < lines.len() && !lines[i].trim_start().starts_with("```") {
                body.push_str(lines[i]); body.push('\n');
                i += 1;
            }
            i += 1;
            let cls = if lang.is_empty() { String::new() } else { format!(" class=\"lang-{}\"", esc(&lang)) };
            out.push_str(&format!("<pre><code{}>{}</code></pre>\n", cls, esc(&body)));
            continue;
        }
        // headings
        if let Some(h) = parse_heading(line) {
            let id = slugify(&h.1);
            let inline = render_inline(&h.1);
            out.push_str(&format!("<h{lv} id=\"{id}\">{txt}</h{lv}>\n", lv=h.0, id=id, txt=inline));
            headings.push(Heading { level: h.0, text: h.1, id });
            i += 1;
            continue;
        }
        // horizontal rule
        if line.trim() == "---" || line.trim() == "***" {
            out.push_str("<hr>\n"); i += 1; continue;
        }
        // blockquote (consecutive > lines)
        if line.starts_with("> ") || line == ">" {
            let mut body = String::new();
            while i < lines.len() && (lines[i].starts_with("> ") || lines[i] == ">") {
                let t = if lines[i] == ">" { "" } else { &lines[i][2..] };
                body.push_str(t); body.push('\n');
                i += 1;
            }
            out.push_str("<blockquote>\n");
            let (inner, _) = render_markdown(&body);
            out.push_str(&inner);
            out.push_str("</blockquote>\n");
            continue;
        }
        // unordered list
        if is_ul_item(line) {
            out.push_str("<ul>\n");
            while i < lines.len() && is_ul_item(lines[i]) {
                let item = strip_ul(lines[i]);
                out.push_str(&format!("<li>{}</li>\n", render_inline(item)));
                i += 1;
            }
            out.push_str("</ul>\n");
            continue;
        }
        // ordered list
        if is_ol_item(line) {
            out.push_str("<ol>\n");
            while i < lines.len() && is_ol_item(lines[i]) {
                let item = strip_ol(lines[i]);
                out.push_str(&format!("<li>{}</li>\n", render_inline(item)));
                i += 1;
            }
            out.push_str("</ol>\n");
            continue;
        }
        // table (pipe style with separator row)
        if is_table_header(&lines, i) {
            let (html, consumed) = render_table(&lines[i..]);
            out.push_str(&html);
            i += consumed;
            continue;
        }
        // blank line
        if line.trim().is_empty() { i += 1; continue; }
        // paragraph — consume consecutive non-blank, non-block-starter lines
        let mut para = String::new();
        while i < lines.len() {
            let l = lines[i];
            if l.trim().is_empty() { break; }
            if l.trim_start().starts_with("```") { break; }
            if parse_heading(l).is_some() { break; }
            if is_ul_item(l) || is_ol_item(l) { break; }
            if l.starts_with("> ") || l == ">" { break; }
            if !para.is_empty() { para.push(' '); }
            para.push_str(l.trim());
            i += 1;
        }
        if !para.is_empty() {
            out.push_str(&format!("<p>{}</p>\n", render_inline(&para)));
        }
    }
    (out, headings)
}

fn parse_heading(line: &str) -> Option<(u8, String)> {
    let t = line.trim_start();
    let mut n: u8 = 0;
    for c in t.chars() { if c == '#' { n += 1; } else { break; } }
    if n == 0 || n > 6 { return None; }
    let rest = &t[n as usize..];
    if !rest.starts_with(' ') { return None; }
    Some((n, rest.trim().to_string()))
}

fn is_ul_item(l: &str) -> bool {
    let t = l.trim_start();
    t.starts_with("- ") || t.starts_with("* ") || t.starts_with("+ ")
}
fn strip_ul(l: &str) -> &str {
    let t = l.trim_start();
    &t[2..]
}
fn is_ol_item(l: &str) -> bool {
    let t = l.trim_start();
    let mut chars = t.chars();
    let mut saw = false;
    while let Some(c) = chars.next() {
        if c.is_ascii_digit() { saw = true; continue; }
        if saw && c == '.' {
            if let Some(sp) = chars.next() { return sp == ' '; }
        }
        return false;
    }
    false
}
fn strip_ol(l: &str) -> &str {
    let t = l.trim_start();
    let dot = t.find('.').unwrap_or(0);
    t[dot+1..].trim_start()
}

fn is_table_header(lines: &[&str], i: usize) -> bool {
    if i + 1 >= lines.len() { return false; }
    let h = lines[i]; let s = lines[i+1];
    if !h.contains('|') || !s.contains('|') { return false; }
    // separator row is all dashes, colons, pipes, spaces
    s.chars().all(|c| matches!(c, '-' | ':' | '|' | ' ')) && s.contains('-')
}

fn render_table(lines: &[&str]) -> (String, usize) {
    let head = split_row(lines[0]);
    let mut html = String::from("<table>\n<thead><tr>");
    for c in &head { html.push_str(&format!("<th>{}</th>", render_inline(c))); }
    html.push_str("</tr></thead>\n<tbody>\n");
    let mut consumed = 2;
    while consumed < lines.len() && lines[consumed].contains('|') && !lines[consumed].trim().is_empty() {
        let cells = split_row(lines[consumed]);
        html.push_str("<tr>");
        for c in &cells { html.push_str(&format!("<td>{}</td>", render_inline(c))); }
        html.push_str("</tr>\n");
        consumed += 1;
    }
    html.push_str("</tbody></table>\n");
    (html, consumed)
}

fn split_row(line: &str) -> Vec<String> {
    let t = line.trim();
    let t = t.trim_start_matches('|').trim_end_matches('|');
    t.split('|').map(|s| s.trim().to_string()).collect()
}

fn wrap_page(body: &str, headings: &[Heading], title: &str, page_title: &str, slug: &str) -> String {
    let toc = build_toc(headings);
    let sidebar = build_sidebar(slug);
    format!(r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#0d1117">
<title>{title} · 1bit.systems docs</title>
<link rel="icon" type="image/svg+xml" href="/assets/logo.svg">
<link rel="stylesheet" href="/assets/style.css">
</head>
<body>
<header class="topbar">
  <a class="brand" href="/"><img src="/assets/logo.svg" alt="" width="28" height="28"><span><b>1</b>bit<span class="dot">.</span>systems</span></a>
  <nav>
    <a href="/docs/">docs</a>
    <a href="/voice/">voice</a>
    <a href="/audio/">audio</a>
    <a href="/join/">join</a>
    <a href="https://github.com/bong-water-water-bong" rel="noopener" class="gh-btn">GitHub</a>
  </nav>
</header>
<div class="docs-shell">
<aside class="docs-side">
{sidebar}
</aside>
<main class="docs-body">
<h1>{page_title}</h1>
{body}
</main>
<aside class="docs-toc">
{toc}
</aside>
</div>
<footer class="footer">
  <p><b>1bit.systems</b> · the 1bit monster · <a href="/">home</a></p>
</footer>
</body>
</html>
"##,
        title = esc(title),
        page_title = esc(page_title),
        sidebar = sidebar,
        body = body,
        toc = toc,
    )
}

fn build_toc(headings: &[Heading]) -> String {
    if headings.iter().filter(|h| h.level >= 2 && h.level <= 3).count() == 0 {
        return String::new();
    }
    let mut s = String::from("<h4>on this page</h4>\n");
    for h in headings {
        if h.level < 2 || h.level > 3 { continue; }
        let indent = if h.level == 3 { " style=\"padding-left:.8rem\"" } else { "" };
        s.push_str(&format!("<a href=\"#{id}\"{ind}>{t}</a>\n", id=h.id, ind=indent, t=esc(&h.text)));
    }
    s
}

// Hard-coded sidebar mirrors docs/wiki layout. Order matches Home.md.
fn build_sidebar(active: &str) -> String {
    let groups: &[(&str, &[(&str, &str)])] = &[
        ("decisions", &[
            ("Why Ternary", "why-ternary"),
            ("Why Rust", "why-rust"),
            ("Why Strix Halo", "why-strix-halo"),
            ("Why Shadow-Burnin", "why-shadow-burnin"),
            ("Why H1b Format", "why-h1b-format"),
            ("Why Caddy + Systemd", "why-caddy-systemd"),
            ("Why halo-agents", "why-halo-agents"),
            ("Why No Python", "why-no-python"),
            ("Why Parity Gates", "why-parity-gates"),
            ("Why No NPU Yet", "why-no-npu-yet"),
            ("Why halo-power", "why-halo-power"),
            ("Why This Way + How", "why-this-way-how"),
        ]),
        ("integrations", &[
            ("Hermes", "hermes-integration"),
            ("AMD GAIA", "amd-gaia-integration"),
            ("Medusa plan", "medusa-integration-plan"),
            ("Rotorquant", "rotorquant-default-decision"),
            ("Whisper streaming", "halo-whisper-streaming-plan"),
            ("NPU kernel", "npu-kernel-design"),
            ("Ternary on AIE", "ternary-on-aie-pack-plan"),
            ("Peak projection", "peak-performance-projection"),
            ("CPU lane", "cpu-lane-plan"),
            ("Cloudflare tunnel", "cloudflare-tunnel-setup"),
            ("VPN-only API", "vpn-only-api"),
            ("Beta 10-day TTL", "beta-10-day-ttl"),
            ("SDD workflow", "sdd-workflow"),
        ]),
        ("crates", &[
            ("halo-echo", "crate-halo-echo"),
            ("halo-helm", "crate-halo-helm"),
            ("halo-landing", "crate-halo-landing"),
            ("halo-lemonade", "crate-halo-lemonade"),
            ("halo-agents-watch", "crate-halo-agents-watch"),
        ]),
        ("amd scans", &[
            ("AI/ML tools", "amd-ai-ml-tools-scan"),
            ("Compilers", "amd-compilers-analyzers-scan"),
            ("Manageability", "amd-manageability-scan"),
            ("Platform driver", "amd-platform-driver-scan"),
            ("ROCm AI hub", "amd-rocm-ai-hub-scan"),
        ]),
        ("reference", &[
            ("Home", "home"),
            ("FAQ", "faq"),
            ("Benchmarks", "benchmarks"),
        ]),
    ];
    let mut s = String::new();
    for (group, items) in groups {
        s.push_str(&format!("<h4>{}</h4>\n", group));
        for (label, slug) in *items {
            let cls = if *slug == active { " class=\"active\"" } else { "" };
            s.push_str(&format!("<a href=\"/docs/{slug}.html\"{cls}>{label}</a>\n",
                slug=slug, cls=cls, label=label));
        }
    }
    s
}
