// build.rs — single-file, std-only markdown → HTML converter for halo-ai.studio/docs
//
// Compile:  rustc -O build.rs -o build_docs
// Run:      ./build_docs <wiki_src_dir> <out_dir>
//
// Handles the subset of markdown actually used in docs/wiki/*.md as of 2026-04-20:
//   - ATX headers (# .. ######)
//   - paragraphs
//   - fenced code blocks (```lang ... ```) with language class
//   - inline code `...`
//   - bold **..**, italic *..*
//   - links [txt](url)
//   - GFM tables with alignment row
//   - unordered lists -/* and ordered lists 1.
//   - blockquotes >
//   - horizontal rules ---
//   - HTML passthrough for inline <...> that already looks like HTML
//
// Links ending in `.md` are rewritten to `.html` (lower-case slug form).
// Links to `../../<file>` (outside the docs tree) are left as text — public deploy
// doesn't ship those files, and a dead link is worse than a pointer.

use std::fs;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: build_docs <wiki_src_dir> <out_dir>");
        std::process::exit(2);
    }
    let src = &args[1];
    let out = &args[2];
    fs::create_dir_all(out).expect("mkdir out");

    // Sidebar grouping — hand-curated, matches the brief.
    let groups: Vec<(&str, Vec<&str>)> = vec![
        ("Getting started", vec!["Home", "FAQ"]),
        (
            "Why we built it this way",
            vec![
                "Why-Ternary",
                "Why-Rust",
                "Why-Strix-Halo",
                "Why-Shadow-Burnin",
                "Why-No-Python",
                "Why-No-NPU-Yet",
                "Why-This-Way-How",
                "Why-halo-power",
            ],
        ),
        (
            "Research",
            vec![
                "Medusa-Integration-Plan",
                "Halo-Whisper-Streaming-Plan",
                "Rotorquant-Default-Decision",
            ],
        ),
        (
            "Integrations",
            vec!["Hermes-Integration", "Cloudflare-Tunnel-Setup"],
        ),
    ];

    let src_dir = Path::new(src);
    let entries: Vec<_> = fs::read_dir(src_dir)
        .expect("read src dir")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|x| x == "md")
                .unwrap_or(false)
        })
        .collect();

    // Map stem -> (slug, title)
    let mut pages: Vec<(String, String, String, String)> = Vec::new(); // (stem, slug, title, body_md)
    for e in &entries {
        let path = e.path();
        let stem = path.file_stem().unwrap().to_string_lossy().to_string();
        let md = fs::read_to_string(&path).expect("read md");
        let title = first_title(&md).unwrap_or_else(|| stem.clone());
        let slug = slugify(&stem);
        pages.push((stem, slug, title, md));
    }
    pages.sort_by(|a, b| a.0.cmp(&b.0));

    // Build sidebar HTML once.
    let sidebar_html = render_sidebar(&groups, &pages);

    // Build search index entries.
    let mut index_entries: Vec<String> = Vec::new();

    for (stem, slug, title, md) in &pages {
        let body_html = md_to_html(md);
        let keywords = extract_keywords(md);
        let page_html = render_page(title, &sidebar_html, &body_html, slug);
        let out_path = Path::new(out).join(format!("{}.html", slug));
        fs::write(&out_path, page_html).expect("write page");

        let entry = format!(
            "{{\"slug\":\"{}\",\"title\":{},\"keywords\":{}}}",
            slug,
            json_str(title),
            json_arr(&keywords),
        );
        index_entries.push(entry);
        let _ = stem; // quiet warning
    }

    // index.json
    let idx = format!("[{}]", index_entries.join(","));
    fs::write(Path::new(out).join("index.json"), idx).expect("write index.json");

    // docs landing (index.html) — grid of cards grouped.
    let landing = render_landing(&groups, &pages, &sidebar_html);
    fs::write(Path::new(out).join("index.html"), landing).expect("write docs index");

    eprintln!("rendered {} pages into {}", pages.len(), out);
}

// ── helpers ─────────────────────────────────────────────────────────

fn slugify(stem: &str) -> String {
    // Filename is already kebab-ish (Hyphen-Separated). Lowercase to canonicalise.
    stem.to_lowercase()
}

fn first_title(md: &str) -> Option<String> {
    for line in md.lines() {
        let t = line.trim_start();
        if let Some(rest) = t.strip_prefix("# ") {
            return Some(rest.trim().to_string());
        }
    }
    None
}

fn extract_keywords(md: &str) -> Vec<String> {
    // Cheap keyword pass: all alpha tokens ≥4 chars, deduped, lower-cased, skip stopwords.
    // Bound to ~80 per doc so index.json stays compact.
    const STOP: &[&str] = &[
        "that", "this", "with", "from", "have", "what", "when", "which", "they", "their", "there",
        "would", "could", "should", "about", "into", "than", "then", "will", "been", "were", "does",
        "only", "also", "other", "every", "some", "most", "more", "much", "very", "like", "just",
        "each", "because", "being", "them", "these", "those", "before", "after", "over", "under",
        "here", "ours", "yours", "your", "mine", "into", "same", "one", "two",
    ];
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    let mut buf = String::new();
    for ch in md.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            buf.push(ch.to_ascii_lowercase());
        } else {
            if buf.len() >= 4 {
                if !STOP.contains(&buf.as_str()) && !buf.chars().all(|c| c.is_ascii_digit()) {
                    if seen.insert(buf.clone()) {
                        out.push(buf.clone());
                        if out.len() >= 80 {
                            buf.clear();
                            break;
                        }
                    }
                }
            }
            buf.clear();
        }
    }
    if buf.len() >= 4 && !STOP.contains(&buf.as_str()) && seen.insert(buf.clone()) {
        out.push(buf);
    }
    out
}

fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn json_arr(items: &[String]) -> String {
    let mut o = String::from("[");
    for (i, s) in items.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        o.push_str(&json_str(s));
    }
    o.push(']');
    o
}

// ── markdown parser (line-level state machine) ──────────────────────

fn md_to_html(md: &str) -> String {
    let lines: Vec<&str> = md.lines().collect();
    let mut out = String::new();
    let mut i = 0;
    let mut in_list: Option<&'static str> = None; // "ul" or "ol"
    let mut in_para = false;
    let mut in_quote = false;

    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim_end();

        // Fenced code block.
        if let Some(lang) = trimmed.trim_start().strip_prefix("```") {
            close_para(&mut out, &mut in_para);
            close_list(&mut out, &mut in_list);
            close_quote(&mut out, &mut in_quote);
            let lang = lang.trim();
            let lang_class = if lang.is_empty() {
                "".to_string()
            } else {
                format!(" class=\"language-{}\"", escape_attr(lang))
            };
            out.push_str(&format!("<pre><code{}>", lang_class));
            i += 1;
            while i < lines.len() {
                let l = lines[i];
                if l.trim_start().starts_with("```") {
                    break;
                }
                out.push_str(&escape_html(l));
                out.push('\n');
                i += 1;
            }
            out.push_str("</code></pre>\n");
            i += 1;
            continue;
        }

        // Header.
        if let Some((level, text)) = parse_header(trimmed) {
            close_para(&mut out, &mut in_para);
            close_list(&mut out, &mut in_list);
            close_quote(&mut out, &mut in_quote);
            // Skip level-1 header — rendered by the page template <h1> already.
            if level > 1 {
                let id = heading_id(text);
                out.push_str(&format!(
                    "<h{lvl} id=\"{id}\">{body}</h{lvl}>\n",
                    lvl = level,
                    id = id,
                    body = inline(text),
                ));
            }
            i += 1;
            continue;
        }

        // Horizontal rule.
        if trimmed == "---" || trimmed == "***" || trimmed == "___" {
            close_para(&mut out, &mut in_para);
            close_list(&mut out, &mut in_list);
            close_quote(&mut out, &mut in_quote);
            out.push_str("<hr>\n");
            i += 1;
            continue;
        }

        // Table: detect header row followed by alignment row.
        if is_table_row(trimmed) && i + 1 < lines.len() && is_table_sep(lines[i + 1].trim_end()) {
            close_para(&mut out, &mut in_para);
            close_list(&mut out, &mut in_list);
            close_quote(&mut out, &mut in_quote);
            let header_cells = split_table_row(trimmed);
            let align_cells = parse_align(lines[i + 1].trim_end());
            out.push_str("<table>\n<thead><tr>");
            for (idx, c) in header_cells.iter().enumerate() {
                let a = align_cells.get(idx).copied().unwrap_or("");
                let style = if a.is_empty() {
                    String::new()
                } else {
                    format!(" style=\"text-align:{}\"", a)
                };
                out.push_str(&format!("<th{}>{}</th>", style, inline(c)));
            }
            out.push_str("</tr></thead>\n<tbody>\n");
            i += 2;
            while i < lines.len() {
                let row = lines[i].trim_end();
                if !is_table_row(row) {
                    break;
                }
                let cells = split_table_row(row);
                out.push_str("<tr>");
                for (idx, c) in cells.iter().enumerate() {
                    let a = align_cells.get(idx).copied().unwrap_or("");
                    let style = if a.is_empty() {
                        String::new()
                    } else {
                        format!(" style=\"text-align:{}\"", a)
                    };
                    out.push_str(&format!("<td{}>{}</td>", style, inline(c)));
                }
                out.push_str("</tr>\n");
                i += 1;
            }
            out.push_str("</tbody></table>\n");
            continue;
        }

        // Blockquote.
        if let Some(rest) = trimmed.strip_prefix("> ").or_else(|| trimmed.strip_prefix(">")) {
            close_para(&mut out, &mut in_para);
            close_list(&mut out, &mut in_list);
            if !in_quote {
                out.push_str("<blockquote>\n");
                in_quote = true;
            }
            out.push_str(&format!("<p>{}</p>\n", inline(rest.trim_start())));
            i += 1;
            continue;
        } else if in_quote {
            close_quote(&mut out, &mut in_quote);
        }

        // Unordered list item.
        if let Some(rest) = strip_ul_marker(trimmed) {
            close_para(&mut out, &mut in_para);
            if in_list != Some("ul") {
                close_list(&mut out, &mut in_list);
                out.push_str("<ul>\n");
                in_list = Some("ul");
            }
            out.push_str(&format!("<li>{}</li>\n", inline(rest)));
            i += 1;
            continue;
        }

        // Ordered list item.
        if let Some(rest) = strip_ol_marker(trimmed) {
            close_para(&mut out, &mut in_para);
            if in_list != Some("ol") {
                close_list(&mut out, &mut in_list);
                out.push_str("<ol>\n");
                in_list = Some("ol");
            }
            out.push_str(&format!("<li>{}</li>\n", inline(rest)));
            i += 1;
            continue;
        }

        // Blank line closes open blocks.
        if trimmed.is_empty() {
            close_para(&mut out, &mut in_para);
            close_list(&mut out, &mut in_list);
            close_quote(&mut out, &mut in_quote);
            i += 1;
            continue;
        }

        // Paragraph: accumulate lines until blank.
        if !in_para {
            out.push_str("<p>");
            in_para = true;
        } else {
            out.push(' ');
        }
        out.push_str(&inline(trimmed));
        i += 1;
    }
    close_para(&mut out, &mut in_para);
    close_list(&mut out, &mut in_list);
    close_quote(&mut out, &mut in_quote);
    out
}

fn close_para(out: &mut String, flag: &mut bool) {
    if *flag {
        out.push_str("</p>\n");
        *flag = false;
    }
}
fn close_list(out: &mut String, flag: &mut Option<&'static str>) {
    if let Some(tag) = *flag {
        out.push_str(&format!("</{}>\n", tag));
        *flag = None;
    }
}
fn close_quote(out: &mut String, flag: &mut bool) {
    if *flag {
        out.push_str("</blockquote>\n");
        *flag = false;
    }
}

fn parse_header(line: &str) -> Option<(usize, &str)> {
    let t = line.trim_start();
    let mut lvl = 0;
    let mut chars = t.chars();
    while let Some('#') = chars.clone().next() {
        chars.next();
        lvl += 1;
        if lvl > 6 {
            return None;
        }
    }
    if lvl == 0 {
        return None;
    }
    let rest = chars.as_str();
    if !rest.starts_with(' ') {
        return None;
    }
    Some((lvl, rest.trim_start()))
}

fn heading_id(text: &str) -> String {
    let mut o = String::new();
    let mut last_dash = false;
    for c in text.chars() {
        let lc = c.to_ascii_lowercase();
        if lc.is_ascii_alphanumeric() {
            o.push(lc);
            last_dash = false;
        } else if !last_dash && !o.is_empty() {
            o.push('-');
            last_dash = true;
        }
    }
    if o.ends_with('-') {
        o.pop();
    }
    o
}

fn is_table_row(line: &str) -> bool {
    let t = line.trim();
    t.starts_with('|') && t.ends_with('|') && t.matches('|').count() >= 2
}

fn is_table_sep(line: &str) -> bool {
    let t = line.trim();
    if !(t.starts_with('|') && t.ends_with('|')) {
        return false;
    }
    let inner = &t[1..t.len() - 1];
    inner
        .split('|')
        .all(|c| {
            let c = c.trim();
            !c.is_empty()
                && c.chars()
                    .all(|ch| ch == '-' || ch == ':' || ch == ' ')
                && c.contains('-')
        })
}

fn split_table_row(line: &str) -> Vec<String> {
    let t = line.trim();
    let inner = &t[1..t.len() - 1];
    inner.split('|').map(|c| c.trim().to_string()).collect()
}

fn parse_align(line: &str) -> Vec<&'static str> {
    let t = line.trim();
    let inner = &t[1..t.len() - 1];
    inner
        .split('|')
        .map(|c| {
            let c = c.trim();
            let l = c.starts_with(':');
            let r = c.ends_with(':');
            match (l, r) {
                (true, true) => "center",
                (false, true) => "right",
                (true, false) => "left",
                _ => "",
            }
        })
        .collect()
}

fn strip_ul_marker(line: &str) -> Option<&str> {
    let t = line.trim_start();
    if let Some(r) = t.strip_prefix("- ") {
        return Some(r);
    }
    if let Some(r) = t.strip_prefix("* ") {
        return Some(r);
    }
    if let Some(r) = t.strip_prefix("+ ") {
        return Some(r);
    }
    None
}

fn strip_ol_marker(line: &str) -> Option<&str> {
    let t = line.trim_start();
    let mut digits = 0;
    for c in t.chars() {
        if c.is_ascii_digit() {
            digits += 1;
        } else {
            break;
        }
    }
    if digits == 0 {
        return None;
    }
    let rest = &t[digits..];
    if let Some(r) = rest.strip_prefix(". ") {
        return Some(r);
    }
    None
}

// ── inline markdown: code, bold, italic, links ──────────────────────

fn inline(src: &str) -> String {
    let bytes = src.as_bytes();
    let mut out = String::new();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        // Non-ASCII byte: copy the whole UTF-8 scalar verbatim and advance past it.
        if b >= 0x80 {
            let ch_len = utf8_char_len(b);
            let end = (i + ch_len).min(bytes.len());
            out.push_str(&src[i..end]);
            i = end;
            continue;
        }
        let c = b as char;

        // Inline code.
        if c == '`' {
            // Count backticks.
            let mut n = 0;
            while i + n < bytes.len() && bytes[i + n] == b'`' {
                n += 1;
            }
            // Find matching run.
            let start = i + n;
            let mut j = start;
            while j < bytes.len() {
                if bytes[j] == b'`' {
                    let mut m = 0;
                    while j + m < bytes.len() && bytes[j + m] == b'`' {
                        m += 1;
                    }
                    if m == n {
                        break;
                    }
                    j += m;
                } else {
                    j += 1;
                }
            }
            if j < bytes.len() {
                let code = &src[start..j];
                out.push_str("<code>");
                out.push_str(&escape_html(code.trim()));
                out.push_str("</code>");
                i = j + n;
                continue;
            }
        }

        // Link [text](url).
        if c == '[' {
            if let Some((text, url, end)) = parse_link(&src[i..]) {
                let final_url = rewrite_link(&url);
                out.push_str(&format!(
                    "<a href=\"{}\">{}</a>",
                    escape_attr(&final_url),
                    inline(&text)
                ));
                i += end;
                continue;
            }
        }

        // Bold **..**.
        if c == '*' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            if let Some(end) = src[i + 2..].find("**") {
                let inner = &src[i + 2..i + 2 + end];
                out.push_str(&format!("<strong>{}</strong>", inline(inner)));
                i += 2 + end + 2;
                continue;
            }
        }

        // Italic *..* (must not be **).
        if c == '*' {
            if let Some(end) = src[i + 1..].find('*') {
                let inner = &src[i + 1..i + 1 + end];
                if !inner.contains('\n') && !inner.is_empty() {
                    out.push_str(&format!("<em>{}</em>", inline(inner)));
                    i += 1 + end + 1;
                    continue;
                }
            }
        }

        // Escape HTML special chars.
        match c {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            c => out.push(c),
        }
        i += 1;
    }
    out
}

fn parse_link(s: &str) -> Option<(String, String, usize)> {
    // s starts with '['
    let bytes = s.as_bytes();
    if bytes[0] != b'[' {
        return None;
    }
    // Find ']('  — allow nested [] inside text (one level).
    let mut depth = 1;
    let mut j = 1;
    while j < bytes.len() {
        let c = bytes[j];
        if c == b'[' {
            depth += 1;
        } else if c == b']' {
            depth -= 1;
            if depth == 0 {
                break;
            }
        }
        j += 1;
    }
    if j >= bytes.len() || bytes[j] != b']' {
        return None;
    }
    if j + 1 >= bytes.len() || bytes[j + 1] != b'(' {
        return None;
    }
    let text = &s[1..j];
    let url_start = j + 2;
    let mut k = url_start;
    let mut paren = 1;
    while k < bytes.len() {
        let c = bytes[k];
        if c == b'(' {
            paren += 1;
        } else if c == b')' {
            paren -= 1;
            if paren == 0 {
                break;
            }
        }
        k += 1;
    }
    if k >= bytes.len() {
        return None;
    }
    let url = &s[url_start..k];
    Some((text.to_string(), url.to_string(), k + 1))
}

fn rewrite_link(url: &str) -> String {
    // External URL untouched.
    if url.starts_with("http://") || url.starts_with("https://") || url.starts_with("mailto:") || url.starts_with("#") {
        return url.to_string();
    }
    // Pointer to repo-local file that won't exist in the public deploy — leave as-is,
    // it'll 404 but preserves the reference for curious readers.
    if url.contains("../../") || url.starts_with("memory-only") {
        return url.to_string();
    }
    // ./Foo-Bar.md or Foo-Bar.md → /docs/foo-bar.html
    let stripped = url.trim_start_matches("./");
    if let Some(base) = stripped.strip_suffix(".md") {
        // Strip any '#frag'
        let (stem, frag) = match base.find('#') {
            Some(p) => (&base[..p], &base[p..]),
            None => (base, ""),
        };
        return format!("/docs/{}.html{}", stem.to_lowercase(), frag);
    }
    url.to_string()
}

/// Render a title string: backticks → <code>, HTML-escape the rest. No links, no bold.
/// Used in sidebar list items, page <title>, and card headings.
fn inline_title(s: &str) -> String {
    let mut out = String::new();
    let mut in_code = false;
    for c in s.chars() {
        if c == '`' {
            if in_code {
                out.push_str("</code>");
            } else {
                out.push_str("<code>");
            }
            in_code = !in_code;
            continue;
        }
        match c {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            c => out.push(c),
        }
    }
    if in_code {
        out.push_str("</code>");
    }
    out
}

fn escape_html(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '<' => o.push_str("&lt;"),
            '>' => o.push_str("&gt;"),
            '&' => o.push_str("&amp;"),
            c => o.push(c),
        }
    }
    o
}

fn utf8_char_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b < 0xC0 {
        1 // continuation byte (shouldn't start a char, but be safe)
    } else if b < 0xE0 {
        2
    } else if b < 0xF0 {
        3
    } else {
        4
    }
}

fn escape_attr(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => o.push_str("&quot;"),
            '<' => o.push_str("&lt;"),
            '>' => o.push_str("&gt;"),
            '&' => o.push_str("&amp;"),
            c => o.push(c),
        }
    }
    o
}

// ── sidebar + page templates ────────────────────────────────────────

fn render_sidebar(
    groups: &[(&str, Vec<&str>)],
    pages: &[(String, String, String, String)],
) -> String {
    let mut s = String::from("<nav class=\"doc-sidebar\" id=\"sidebar\">\n");
    s.push_str("<a class=\"home-link\" href=\"/docs/\">&larr; all docs</a>\n");
    for (group, stems) in groups {
        s.push_str(&format!("<div class=\"group\"><h4>{}</h4><ul>\n", group));
        for stem in stems {
            if let Some((_, slug, title, _)) = pages.iter().find(|(st, _, _, _)| st == stem) {
                s.push_str(&format!(
                    "<li data-slug=\"{}\"><a href=\"/docs/{}.html\">{}</a></li>\n",
                    slug,
                    slug,
                    inline_title(title)
                ));
            }
        }
        s.push_str("</ul></div>\n");
    }
    // Any pages not in a group go into an "Other" bucket so nothing silently drops.
    let grouped: std::collections::BTreeSet<&str> =
        groups.iter().flat_map(|(_, v)| v.iter().copied()).collect();
    let orphans: Vec<_> = pages
        .iter()
        .filter(|(stem, _, _, _)| !grouped.contains(stem.as_str()))
        .collect();
    if !orphans.is_empty() {
        s.push_str("<div class=\"group\"><h4>Other</h4><ul>\n");
        for (_, slug, title, _) in orphans {
            s.push_str(&format!(
                "<li data-slug=\"{}\"><a href=\"/docs/{}.html\">{}</a></li>\n",
                slug,
                slug,
                inline_title(title)
            ));
        }
        s.push_str("</ul></div>\n");
    }
    s.push_str("</nav>\n");
    s
}

fn render_page(title: &str, sidebar: &str, body: &str, slug: &str) -> String {
    format!(
        "<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
  <title>{t} · halo-ai docs</title>
  <meta name=\"description\" content=\"halo-ai docs: {t_desc}\">
  <link rel=\"stylesheet\" href=\"/docs/docs.css\">
</head>
<body data-slug=\"{slug}\">
<nav class=\"doc-topbar\">
  <a href=\"/\" class=\"logo\">halo-<span>ai</span></a>
  <a href=\"/docs/\">docs</a>
  <input type=\"search\" placeholder=\"filter…\" id=\"q\" autocomplete=\"off\">
</nav>
<div class=\"doc-shell\">
{sidebar}
<main class=\"doc-main\">
  <h1>{t_html}</h1>
  {body}
</main>
</div>
<script src=\"/docs/docs.js\"></script>
</body>
</html>
",
        t = escape_html(title),
        t_html = inline_title(title),
        t_desc = escape_attr(title),
        slug = slug,
        sidebar = sidebar,
        body = body,
    )
}

fn render_landing(
    groups: &[(&str, Vec<&str>)],
    pages: &[(String, String, String, String)],
    sidebar: &str,
) -> String {
    let mut cards = String::new();
    for (group, stems) in groups {
        cards.push_str(&format!(
            "<section class=\"doc-group\"><h2>{}</h2><div class=\"doc-cards\">\n",
            escape_html(group)
        ));
        for stem in stems {
            if let Some((_, slug, title, md)) =
                pages.iter().find(|(st, _, _, _)| st == stem)
            {
                let blurb = first_paragraph(md);
                cards.push_str(&format!(
                    "<a class=\"doc-card\" href=\"/docs/{}.html\"><h3>{}</h3><p>{}</p></a>\n",
                    slug,
                    inline_title(title),
                    escape_html(&blurb)
                ));
            }
        }
        cards.push_str("</div></section>\n");
    }
    // Other bucket.
    let grouped: std::collections::BTreeSet<&str> =
        groups.iter().flat_map(|(_, v)| v.iter().copied()).collect();
    let orphans: Vec<_> = pages
        .iter()
        .filter(|(stem, _, _, _)| !grouped.contains(stem.as_str()))
        .collect();
    if !orphans.is_empty() {
        cards.push_str("<section class=\"doc-group\"><h2>Other</h2><div class=\"doc-cards\">\n");
        for (_, slug, title, md) in orphans {
            let blurb = first_paragraph(md);
            cards.push_str(&format!(
                "<a class=\"doc-card\" href=\"/docs/{}.html\"><h3>{}</h3><p>{}</p></a>\n",
                slug,
                inline_title(title),
                escape_html(&blurb)
            ));
        }
        cards.push_str("</div></section>\n");
    }

    format!(
        "<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
  <title>halo-ai docs</title>
  <meta name=\"description\" content=\"halo-ai docs — why and how the 1-bit inference engine is built. Wiki, research, integrations.\">
  <link rel=\"stylesheet\" href=\"/docs/docs.css\">
</head>
<body data-slug=\"index\">
<nav class=\"doc-topbar\">
  <a href=\"/\" class=\"logo\">halo-<span>ai</span></a>
  <a href=\"/docs/\" class=\"active\">docs</a>
  <input type=\"search\" placeholder=\"filter…\" id=\"q\" autocomplete=\"off\">
</nav>
<div class=\"doc-shell\">
{sidebar}
<main class=\"doc-main doc-landing\">
  <header class=\"doc-hero\">
    <p class=\"eyebrow\">docs</p>
    <h1>how halo-ai works, and why</h1>
    <p class=\"lede\">Every architectural call, every integration path, every benchmark. Use the filter to scope down.</p>
  </header>
  {cards}
</main>
</div>
<script src=\"/docs/docs.js\"></script>
</body>
</html>
",
        sidebar = sidebar,
        cards = cards,
    )
}

fn first_paragraph(md: &str) -> String {
    let mut started = false;
    let mut out = String::new();
    for line in md.lines() {
        let t = line.trim();
        if t.is_empty() {
            if started {
                break;
            } else {
                continue;
            }
        }
        if t.starts_with('#') || t.starts_with("```") || t.starts_with('|') {
            if started {
                break;
            }
            continue;
        }
        if !out.is_empty() {
            out.push(' ');
        }
        // Strip inline markup cheaply.
        let cleaned: String = strip_md(t);
        out.push_str(&cleaned);
        started = true;
    }
    if out.len() > 240 {
        let mut cut = 240;
        while !out.is_char_boundary(cut) {
            cut -= 1;
        }
        out.truncate(cut);
        out.push('…');
    }
    // Trim leading "One-line answer:" marker for cleaner cards.
    if let Some(rest) = out.strip_prefix("One-line answer: ") {
        return rest.to_string();
    }
    out
}

fn strip_md(s: &str) -> String {
    let mut o = String::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '*' | '_' | '`' => continue,
            '[' => {
                // [text](url) → text
                let mut buf = String::new();
                while let Some(&ch) = chars.peek() {
                    if ch == ']' {
                        chars.next();
                        break;
                    }
                    buf.push(chars.next().unwrap());
                }
                // skip (url)
                if chars.peek() == Some(&'(') {
                    chars.next();
                    while let Some(&ch) = chars.peek() {
                        chars.next();
                        if ch == ')' {
                            break;
                        }
                    }
                }
                o.push_str(&buf);
            }
            c => o.push(c),
        }
    }
    o
}
