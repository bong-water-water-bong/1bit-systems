//! halo-gaia — ratatui terminal chat client against halo-server.
//!
//! v0 scope:
//! - Single-pane conversation view + input box + status line.
//! - Enter sends; Ctrl-L / Ctrl-N clear; Ctrl-C / Ctrl-D exit.
//! - Streams the assistant reply via [`GaiaClient::send_stream`] so the user
//!   sees tokens land live. On stream error, falls back silently to showing
//!   the error as the assistant turn so the UI never wedges.
//!
//! Deliberately non-persistent: one ephemeral conversation per process.
//! Config is via env:
//!   HALO_GAIA_URL   — server URL (default http://127.0.0.1:8180)
//!   HALO_GAIA_MODEL — model id (default halo-1bit-2b)
//!   HALO_GAIA_TOKEN — optional bearer token

use std::io::{self, Stdout};
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyModifiers,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use futures::StreamExt;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};

use halo_gaia::{Conversation, GaiaClient, Role, SessionConfig};

// ---- messaging between the stream task and the UI loop --------------------

enum StreamMsg {
    Delta(String),
    Done,
    Error(String),
}

// ---- app state ------------------------------------------------------------

struct App {
    conv: Conversation,
    input: String,
    /// Assistant message currently streaming in; `None` when idle.
    streaming: Option<String>,
    /// Token count across the whole conversation (approx — whitespace-split).
    total_tokens: usize,
    /// Rolling tok/s of the last completed reply.
    last_tps: f32,
    /// Start time of the active stream, for tok/s calc.
    stream_started: Option<Instant>,
    stream_tokens: usize,
    rx: Option<UnboundedReceiver<StreamMsg>>,
    status_err: Option<String>,
}

impl App {
    fn new() -> Self {
        Self {
            conv: Conversation::new(),
            input: String::new(),
            streaming: None,
            total_tokens: 0,
            last_tps: 0.0,
            stream_started: None,
            stream_tokens: 0,
            rx: None,
            status_err: None,
        }
    }

    fn reset(&mut self) {
        self.conv = Conversation::new();
        self.input.clear();
        self.streaming = None;
        self.total_tokens = 0;
        self.last_tps = 0.0;
        self.stream_started = None;
        self.stream_tokens = 0;
        self.rx = None;
        self.status_err = None;
    }

    fn is_streaming(&self) -> bool {
        self.streaming.is_some()
    }
}

// ---- entry point ----------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    let cfg = load_config();
    let client = GaiaClient::new(cfg.clone());

    let mut terminal = enter_tui()?;
    let res = run(&mut terminal, &client, &cfg).await;
    exit_tui(&mut terminal)?;
    res
}

fn load_config() -> SessionConfig {
    let url = std::env::var("HALO_GAIA_URL").unwrap_or_else(|_| "http://127.0.0.1:8180".into());
    let model = std::env::var("HALO_GAIA_MODEL").unwrap_or_else(|_| "halo-1bit-2b".into());
    let mut cfg = SessionConfig::new(url, model);
    cfg.bearer = std::env::var("HALO_GAIA_TOKEN").ok();
    cfg
}

type Tui = Terminal<CrosstermBackend<Stdout>>;

fn enter_tui() -> Result<Tui> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    Ok(Terminal::new(CrosstermBackend::new(stdout))?)
}

fn exit_tui(terminal: &mut Tui) -> Result<()> {
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;
    Ok(())
}

async fn run(terminal: &mut Tui, client: &GaiaClient, cfg: &SessionConfig) -> Result<()> {
    let mut app = App::new();

    loop {
        terminal.draw(|f| draw(f, &app, cfg))?;

        // Drain any pending stream messages without blocking the input loop.
        drain_stream(&mut app);

        // Poll input with a short timeout so the UI refreshes while streaming.
        if event::poll(Duration::from_millis(30))? {
            if let Event::Key(key) = event::read()? {
                match handle_key(&mut app, key, client) {
                    KeyAction::Exit => break,
                    KeyAction::Continue => {}
                }
            }
        }
    }
    Ok(())
}

fn drain_stream(app: &mut App) {
    // Take the rx out to avoid borrow clash while we mutate other fields.
    let Some(mut rx) = app.rx.take() else {
        return;
    };
    let mut done = false;
    loop {
        match rx.try_recv() {
            Ok(StreamMsg::Delta(s)) => {
                if let Some(buf) = app.streaming.as_mut() {
                    buf.push_str(&s);
                }
                app.stream_tokens += 1;
            }
            Ok(StreamMsg::Done) => {
                done = true;
                break;
            }
            Ok(StreamMsg::Error(e)) => {
                app.status_err = Some(e);
                done = true;
                break;
            }
            Err(mpsc::error::TryRecvError::Empty) => break,
            Err(mpsc::error::TryRecvError::Disconnected) => {
                done = true;
                break;
            }
        }
    }
    if done {
        if let (Some(buf), Some(started)) = (app.streaming.take(), app.stream_started.take()) {
            let elapsed = started.elapsed().as_secs_f32().max(0.001);
            app.last_tps = app.stream_tokens as f32 / elapsed;
            app.total_tokens += approx_tokens(&buf);
            app.conv.push_assistant(buf);
            app.stream_tokens = 0;
        }
    } else {
        app.rx = Some(rx);
    }
}

enum KeyAction {
    Continue,
    Exit,
}

fn handle_key(app: &mut App, key: KeyEvent, client: &GaiaClient) -> KeyAction {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    match (key.code, ctrl) {
        (KeyCode::Char('c'), true) | (KeyCode::Char('d'), true) => return KeyAction::Exit,
        (KeyCode::Char('l'), true) | (KeyCode::Char('n'), true) => {
            app.reset();
        }
        (KeyCode::Enter, _) => {
            if app.is_streaming() || app.input.trim().is_empty() {
                return KeyAction::Continue;
            }
            let text = std::mem::take(&mut app.input);
            app.total_tokens += approx_tokens(&text);
            app.conv.push_user(text);
            spawn_stream(app, client);
        }
        (KeyCode::Backspace, _) => {
            app.input.pop();
        }
        (KeyCode::Char(c), false) => {
            app.input.push(c);
        }
        _ => {}
    }
    KeyAction::Continue
}

fn spawn_stream(app: &mut App, client: &GaiaClient) {
    let (tx, rx) = mpsc::unbounded_channel();
    app.rx = Some(rx);
    app.streaming = Some(String::new());
    app.stream_started = Some(Instant::now());
    app.stream_tokens = 0;
    app.status_err = None;

    let stream = client.send_stream(&app.conv);
    tokio::spawn(async move {
        futures::pin_mut!(stream);
        let mut got_any = false;
        while let Some(item) = stream.next().await {
            match item {
                Ok(delta) => {
                    got_any = true;
                    let _ = tx.send(StreamMsg::Delta(delta));
                }
                Err(e) => {
                    send_err(&tx, got_any, e.to_string());
                    return;
                }
            }
        }
        let _ = tx.send(StreamMsg::Done);
    });
}

fn send_err(tx: &UnboundedSender<StreamMsg>, got_any: bool, msg: String) {
    if got_any {
        let _ = tx.send(StreamMsg::Done);
    } else {
        let _ = tx.send(StreamMsg::Error(msg));
    }
}

// ---- rendering ------------------------------------------------------------

fn draw(f: &mut ratatui::Frame, app: &App, cfg: &SessionConfig) {
    let area = f.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(3),    // conversation pane
            Constraint::Length(3), // input box
            Constraint::Length(1), // status line
        ])
        .split(area);

    draw_conversation(f, chunks[0], app);
    draw_input(f, chunks[1], app);
    draw_status(f, chunks[2], app, cfg);
}

fn draw_conversation(f: &mut ratatui::Frame, area: Rect, app: &App) {
    let mut lines: Vec<Line> = Vec::new();
    for turn in &app.conv.turns {
        if matches!(turn.role, Role::System) {
            continue;
        }
        lines.push(role_header(turn.role));
        for line in turn.content.lines() {
            lines.push(Line::from(line.to_owned()));
        }
        lines.push(Line::from(""));
    }
    if let Some(partial) = &app.streaming {
        lines.push(role_header(Role::Assistant));
        for line in partial.lines() {
            lines.push(Line::from(line.to_owned()));
        }
        // Blinking cursor-ish caret so the user knows it's live.
        lines.push(Line::from(Span::styled(
            "_",
            Style::default().add_modifier(Modifier::SLOW_BLINK),
        )));
    }

    let title = " halo-gaia ";
    // Scroll so the newest content is always visible — show the bottom of the
    // buffer by computing a simple offset.
    let visible = area.height.saturating_sub(2) as usize;
    let total = lines.len();
    let scroll = total.saturating_sub(visible) as u16;

    let para = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(title))
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0));
    f.render_widget(para, area);
}

fn role_header(role: Role) -> Line<'static> {
    let (label, color) = match role {
        Role::User => ("User", Color::Cyan),
        Role::Assistant => ("Assistant", Color::Green),
        Role::System => ("System", Color::DarkGray),
    };
    Line::from(Span::styled(
        format!("{label}:"),
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    ))
}

fn draw_input(f: &mut ratatui::Frame, area: Rect, app: &App) {
    let content = if app.is_streaming() {
        Line::from(Span::styled(
            "  (streaming… press Ctrl-C to exit)",
            Style::default().fg(Color::DarkGray),
        ))
    } else {
        Line::from(vec![
            Span::styled("> ", Style::default().fg(Color::Yellow)),
            Span::raw(app.input.clone()),
            Span::styled("_", Style::default().add_modifier(Modifier::SLOW_BLINK)),
        ])
    };
    let para = Paragraph::new(content).block(Block::default().borders(Borders::ALL));
    f.render_widget(para, area);
}

fn draw_status(f: &mut ratatui::Frame, area: Rect, app: &App, cfg: &SessionConfig) {
    let text = if let Some(err) = &app.status_err {
        format!(" ERROR: {err}")
    } else {
        format!(
            " {} · {} · {:.1} tok/s · {} ctx",
            cfg.server_url, cfg.default_model, app.last_tps, app.total_tokens
        )
    };
    let style = if app.status_err.is_some() {
        Style::default().fg(Color::Red)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    let para = Paragraph::new(Line::from(Span::styled(text, style)));
    f.render_widget(para, area);
}

/// Very rough token estimator — whitespace-split word count. Good enough for a
/// status-line counter; the server is the source of truth for real billing.
fn approx_tokens(s: &str) -> usize {
    s.split_whitespace().count()
}
