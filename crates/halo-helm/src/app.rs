//! eframe/egui `App` impl for halo-helm.
//!
//! Five panes, left-nav + top-bar switcher:
//!
//! * Status  — halo-server reachability + loaded model + tok/s readout.
//! * Chat    — single input + assistant bubble list, hits
//!             `POST /v1/chat/completions` with `stream: true`.
//! * Skills  — `halo_agents::SkillStore::list`.
//! * Memory  — `halo_agents::MemoryStore::list(Memory)` + `list(User)`.
//! * Models  — `GET /v1/models` off the server.
//!
//! Persistence: last-open pane + window size save via eframe's `persistence`
//! feature (ron-backed storage in the user's config dir).
//!
//! This module does NOT spin up its own tokio runtime, does NOT open the
//! window, and does NOT do network I/O inside `new()`. Tests construct the
//! app and assert defaults without pulling in eframe's native stack.

use crate::{Conversation, Role, SessionConfig};
use serde::{Deserialize, Serialize};

/// Which pane is currently focused.
///
/// Persisted across restarts via eframe's storage (see `save`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Pane {
    Status,
    Chat,
    Skills,
    Memory,
    Models,
}

impl Pane {
    pub const ALL: &'static [Pane] = &[
        Pane::Status,
        Pane::Chat,
        Pane::Skills,
        Pane::Memory,
        Pane::Models,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Pane::Status => "Status",
            Pane::Chat => "Chat",
            Pane::Skills => "Skills",
            Pane::Memory => "Memory",
            Pane::Models => "Models",
        }
    }
}

impl Default for Pane {
    fn default() -> Self {
        Pane::Status
    }
}

/// Per-skill row rendered in the Skills pane. Name + category + short
/// description. The underlying `halo_agents::Skill` carries more; helm only
/// needs what fits on one line for now.
#[derive(Debug, Clone)]
pub struct SkillRow {
    pub name: String,
    pub category: String,
    pub description: String,
}

/// Snapshot of the halo-server's `/v1/models` list. Empty until refreshed.
#[derive(Debug, Clone, Default)]
pub struct ModelsSnapshot {
    pub ids: Vec<String>,
    pub error: Option<String>,
}

/// Status-pane readout. All stubs for now — Status pane just needs the
/// shape so the egui code compiles; a future refresh thread fills these in.
#[derive(Debug, Clone)]
pub struct StatusReadout {
    pub reachable: bool,
    pub active_model: String,
    pub tok_per_s: f32,
}

impl Default for StatusReadout {
    fn default() -> Self {
        Self {
            reachable: false,
            active_model: "unknown".to_string(),
            tok_per_s: 0.0,
        }
    }
}

/// Top-level eframe app. The eframe-facing `App` impl lives below; this
/// struct itself is plain data so construction + default-state tests don't
/// require an open window.
pub struct HelmApp {
    /// Server + model + bearer config. Loaded from env via `SessionConfig::from_env`.
    pub cfg: SessionConfig,
    /// Pane currently visible in the right-hand content area.
    pub current_pane: Pane,
    /// Conversation buffer for the Chat pane.
    pub chat_conv: Conversation,
    /// User's current half-typed input.
    pub chat_input: String,
    /// Streaming assistant reply, populated by the network worker. None when idle.
    pub chat_streaming: Option<String>,
    /// Latest /v1/models snapshot for the Models pane.
    pub models: ModelsSnapshot,
    /// Latest Skills snapshot for the Skills pane.
    pub skills: Vec<SkillRow>,
    /// Latest MEMORY.md + USER.md entries for the Memory pane.
    pub memory_entries: Vec<String>,
    pub user_entries: Vec<String>,
    /// Status-pane readout. Refreshed by a background probe (future work).
    pub status: StatusReadout,
    /// Last error string shown in the status bar. Rendered in red on the
    /// bottom strip when set; clears on next successful action.
    pub last_error: Option<String>,
}

impl HelmApp {
    /// Build a fresh app from env-derived config. Safe to call outside
    /// eframe — no network, no window, no filesystem I/O.
    pub fn new(cfg: SessionConfig) -> Self {
        Self {
            cfg,
            current_pane: Pane::default(),
            chat_conv: Conversation::new(),
            chat_input: String::new(),
            chat_streaming: None,
            models: ModelsSnapshot::default(),
            skills: Vec::new(),
            memory_entries: Vec::new(),
            user_entries: Vec::new(),
            status: StatusReadout::default(),
            last_error: None,
        }
    }

    /// eframe entry point. Pulls persisted pane + window size out of
    /// `cc.storage` if available.
    pub fn from_cc(cc: &eframe::CreationContext<'_>, cfg: SessionConfig) -> Self {
        let mut app = Self::new(cfg);
        if let Some(storage) = cc.storage {
            if let Some(pane) = eframe::get_value::<Pane>(storage, PERSIST_PANE_KEY) {
                app.current_pane = pane;
            }
        }
        app
    }

    /// Load skills from the shared on-disk `~/.halo/skills` root. Called
    /// on-demand when the Skills pane opens. Errors surface in `last_error`
    /// and leave the prior list intact.
    pub fn refresh_skills(&mut self) {
        match halo_agents::SkillStore::new().and_then(|s| s.list()) {
            Ok(skills) => {
                self.skills = skills
                    .into_iter()
                    .map(|s| SkillRow {
                        name: s.name,
                        category: s.metadata_halo.category,
                        description: s.description,
                    })
                    .collect();
                self.last_error = None;
            }
            Err(e) => self.last_error = Some(format!("skills: {e}")),
        }
    }

    /// Load MEMORY.md + USER.md entries from the shared on-disk root.
    pub fn refresh_memory(&mut self) {
        match halo_agents::MemoryStore::new() {
            Ok(store) => {
                match store.list(halo_agents::MemoryKind::Memory) {
                    Ok(v) => self.memory_entries = v,
                    Err(e) => self.last_error = Some(format!("memory: {e}")),
                }
                match store.list(halo_agents::MemoryKind::User) {
                    Ok(v) => self.user_entries = v,
                    Err(e) => self.last_error = Some(format!("memory user: {e}")),
                }
            }
            Err(e) => self.last_error = Some(format!("memory store: {e}")),
        }
    }
}

/// Storage key for the currently-selected pane.
const PERSIST_PANE_KEY: &str = "helm.current_pane";

// --- eframe wiring ---------------------------------------------------------

impl eframe::App for HelmApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Top bar: name + pane switcher buttons.
        egui::TopBottomPanel::top("helm-top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("halo-helm");
                ui.separator();
                for p in Pane::ALL {
                    let selected = self.current_pane == *p;
                    if ui.selectable_label(selected, p.label()).clicked() {
                        self.current_pane = *p;
                    }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(&self.cfg.server_url);
                });
            });
        });

        // Left nav: same 5 panes (radio-style) + quick config readout.
        egui::SidePanel::left("helm-nav")
            .default_width(160.0)
            .show(ctx, |ui| {
                ui.add_space(6.0);
                ui.label(egui::RichText::new("Panes").strong());
                for p in Pane::ALL {
                    ui.radio_value(&mut self.current_pane, *p, p.label());
                }
                ui.separator();
                ui.label(egui::RichText::new("Model").strong());
                ui.monospace(&self.cfg.default_model);
            });

        // Bottom status strip.
        egui::TopBottomPanel::bottom("helm-status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if let Some(err) = &self.last_error {
                    ui.colored_label(egui::Color32::RED, err);
                } else {
                    ui.label(format!(
                        "{} | {} | {:.1} tok/s",
                        if self.status.reachable { "OK" } else { "unreachable" },
                        self.status.active_model,
                        self.status.tok_per_s,
                    ));
                }
            });
        });

        // Main content.
        egui::CentralPanel::default().show(ctx, |ui| match self.current_pane {
            Pane::Status => draw_status(ui, self),
            Pane::Chat => draw_chat(ui, self),
            Pane::Skills => draw_skills(ui, self),
            Pane::Memory => draw_memory(ui, self),
            Pane::Models => draw_models(ui, self),
        });
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, PERSIST_PANE_KEY, &self.current_pane);
    }
}

// --- pane renderers --------------------------------------------------------
//
// Each pane takes `&mut egui::Ui` + `&mut HelmApp` so it can mutate state in
// response to widget clicks / input. No async here — network work is spawned
// on the ambient tokio runtime from `main` and messages back via channels.

fn draw_status(ui: &mut egui::Ui, app: &HelmApp) {
    ui.heading("Status");
    ui.add_space(8.0);
    ui.horizontal(|ui| {
        ui.label("halo-server:");
        ui.monospace(&app.cfg.server_url);
    });
    ui.horizontal(|ui| {
        ui.label("reachable:");
        if app.status.reachable {
            ui.colored_label(egui::Color32::from_rgb(0, 160, 0), "yes");
        } else {
            ui.colored_label(egui::Color32::from_rgb(160, 0, 0), "unknown");
        }
    });
    ui.horizontal(|ui| {
        ui.label("loaded model:");
        ui.monospace(&app.status.active_model);
    });
    ui.horizontal(|ui| {
        ui.label("tok/s:");
        ui.monospace(format!("{:.1}", app.status.tok_per_s));
    });
    ui.separator();
    ui.label("(placeholder — live probe lands in a follow-up change.)");
}

fn draw_chat(ui: &mut egui::Ui, app: &mut HelmApp) {
    ui.heading("Chat");
    ui.add_space(4.0);

    // Conversation scroll area.
    egui::ScrollArea::vertical()
        .stick_to_bottom(true)
        .max_height(ui.available_height() - 60.0)
        .show(ui, |ui| {
            for turn in &app.chat_conv.turns {
                if matches!(turn.role, Role::System) {
                    continue;
                }
                let (label, color) = match turn.role {
                    Role::User => ("you", egui::Color32::from_rgb(0, 140, 200)),
                    Role::Assistant => ("halo", egui::Color32::from_rgb(0, 160, 0)),
                    Role::System => ("sys", egui::Color32::GRAY),
                };
                ui.colored_label(color, format!("{label}:"));
                ui.label(&turn.content);
                ui.add_space(6.0);
            }
            if let Some(partial) = &app.chat_streaming {
                ui.colored_label(egui::Color32::from_rgb(0, 160, 0), "halo:");
                ui.label(partial);
            }
        });

    ui.separator();
    // Input box + send.
    ui.horizontal(|ui| {
        let send = ui
            .add(
                egui::TextEdit::singleline(&mut app.chat_input)
                    .desired_width(ui.available_width() - 80.0)
                    .hint_text("message…"),
            )
            .lost_focus()
            && ui.input(|i| i.key_pressed(egui::Key::Enter));
        let click = ui.button("Send").clicked();
        if (send || click) && !app.chat_input.trim().is_empty() {
            // For this scaffold we just stash the user turn + set an error
            // placeholder. The real streaming path wires into HelmClient in a
            // follow-up — we want the window + panes to ship first.
            let text = std::mem::take(&mut app.chat_input);
            app.chat_conv.push_user(text);
            app.last_error = Some(
                "chat streaming not wired yet — transport lives in HelmClient; UI wiring next."
                    .into(),
            );
        }
    });
}

fn draw_skills(ui: &mut egui::Ui, app: &mut HelmApp) {
    ui.heading("Skills");
    ui.horizontal(|ui| {
        if ui.button("Refresh").clicked() {
            app.refresh_skills();
        }
        ui.label(format!("{} skill(s)", app.skills.len()));
    });
    ui.separator();
    egui::ScrollArea::vertical().show(ui, |ui| {
        if app.skills.is_empty() {
            ui.label("(no skills loaded — click Refresh)");
            return;
        }
        for s in &app.skills {
            ui.horizontal(|ui| {
                ui.monospace(&s.name);
                ui.label(format!("[{}]", s.category));
            });
            ui.label(&s.description);
            ui.add_space(4.0);
        }
    });
}

fn draw_memory(ui: &mut egui::Ui, app: &mut HelmApp) {
    ui.heading("Memory");
    ui.horizontal(|ui| {
        if ui.button("Refresh").clicked() {
            app.refresh_memory();
        }
        ui.label(format!(
            "{} memory / {} user entries",
            app.memory_entries.len(),
            app.user_entries.len()
        ));
    });
    ui.separator();
    egui::ScrollArea::vertical().show(ui, |ui| {
        ui.label(egui::RichText::new("MEMORY.md").strong());
        if app.memory_entries.is_empty() {
            ui.label("(empty)");
        } else {
            for (i, e) in app.memory_entries.iter().enumerate() {
                ui.label(format!("{}. {}", i + 1, e));
                ui.add_space(2.0);
            }
        }
        ui.separator();
        ui.label(egui::RichText::new("USER.md").strong());
        if app.user_entries.is_empty() {
            ui.label("(empty)");
        } else {
            for (i, e) in app.user_entries.iter().enumerate() {
                ui.label(format!("{}. {}", i + 1, e));
                ui.add_space(2.0);
            }
        }
    });
}

fn draw_models(ui: &mut egui::Ui, app: &mut HelmApp) {
    ui.heading("Models");
    ui.horizontal(|ui| {
        if ui.button("Refresh").clicked() {
            // Populated by a follow-up change — the transport plumbing lives
            // in `HelmClient` and the spawn path needs the runtime handle.
            app.last_error = Some("models refresh not wired yet".into());
        }
        ui.label(format!("{} model(s)", app.models.ids.len()));
    });
    ui.separator();
    if let Some(err) = &app.models.error {
        ui.colored_label(egui::Color32::RED, err);
    }
    egui::ScrollArea::vertical().show(ui, |ui| {
        if app.models.ids.is_empty() {
            ui.label("(no models loaded — click Refresh)");
            return;
        }
        for id in &app.models.ids {
            ui.monospace(id);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> SessionConfig {
        SessionConfig::new("http://127.0.0.1:8180", "halo-1bit-2b")
    }

    #[test]
    fn new_constructs_without_panic_and_defaults_are_sane() {
        // The whole point of keeping `HelmApp::new` runtime-free: no window,
        // no tokio, no filesystem access. If this test hangs or panics, the
        // scaffold is doing too much in `new`.
        let app = HelmApp::new(cfg());
        assert_eq!(app.cfg.server_url, "http://127.0.0.1:8180");
        assert_eq!(app.cfg.default_model, "halo-1bit-2b");
        assert!(app.chat_conv.turns.is_empty());
        assert!(app.chat_input.is_empty());
        assert!(app.chat_streaming.is_none());
        assert!(app.skills.is_empty());
        assert!(app.memory_entries.is_empty());
        assert!(app.user_entries.is_empty());
        assert!(app.models.ids.is_empty());
        assert!(app.last_error.is_none());
    }

    #[test]
    fn default_pane_is_status() {
        // Landing on Status matches the "ops-glance first" intent: a user
        // whose server is down should see the red indicator immediately,
        // not have to click through a Chat pane that won't work.
        assert_eq!(Pane::default(), Pane::Status);
        let app = HelmApp::new(cfg());
        assert_eq!(app.current_pane, Pane::Status);
    }

    #[test]
    fn pane_all_covers_five_panes_in_declared_order() {
        // The top-bar + left-nav iterate `Pane::ALL` to render tabs. If a
        // pane is added to the enum but missed in ALL, the UI silently drops
        // it — this test is the tripwire.
        assert_eq!(Pane::ALL.len(), 5);
        assert_eq!(Pane::ALL[0], Pane::Status);
        assert_eq!(Pane::ALL[1], Pane::Chat);
        assert_eq!(Pane::ALL[2], Pane::Skills);
        assert_eq!(Pane::ALL[3], Pane::Memory);
        assert_eq!(Pane::ALL[4], Pane::Models);
    }

    #[test]
    fn pane_labels_are_stable_strings() {
        // Pane labels are shown in the top bar AND persisted via eframe's
        // storage as the enum discriminant — but the string labels are what
        // the user actually sees. Lock them.
        assert_eq!(Pane::Status.label(), "Status");
        assert_eq!(Pane::Chat.label(), "Chat");
        assert_eq!(Pane::Skills.label(), "Skills");
        assert_eq!(Pane::Memory.label(), "Memory");
        assert_eq!(Pane::Models.label(), "Models");
    }

    #[test]
    fn pane_round_trips_through_serde() {
        // Persistence uses serde → ron (eframe `persistence` feature). If
        // the enum loses a Serialize impl, `save()` silently drops the
        // pane. Assert we can round-trip.
        for p in Pane::ALL {
            let s = serde_json::to_string(p).unwrap();
            let back: Pane = serde_json::from_str(&s).unwrap();
            assert_eq!(back, *p);
        }
    }
}
