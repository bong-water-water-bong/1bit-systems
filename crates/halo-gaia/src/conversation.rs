//! Chat turn buffer + conversion to the OpenAI `messages` wire shape.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatTurn {
    pub role: Role,
    pub content: String,
    pub ts: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Conversation {
    pub turns: Vec<ChatTurn>,
}

impl Conversation {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_system(&mut self, s: String) {
        self.push(Role::System, s);
    }
    pub fn push_user(&mut self, s: String) {
        self.push(Role::User, s);
    }
    pub fn push_assistant(&mut self, s: String) {
        self.push(Role::Assistant, s);
    }

    fn push(&mut self, role: Role, content: String) {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        self.turns.push(ChatTurn { role, content, ts });
    }

    /// Render as the array shape OpenAI `/v1/chat/completions` expects.
    pub fn to_openai_messages(&self) -> Vec<Value> {
        self.turns
            .iter()
            .map(|t| json!({ "role": t.role.to_string(), "content": t.content }))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_display_matches_openai_strings() {
        assert_eq!(Role::System.to_string(), "system");
        assert_eq!(Role::User.to_string(), "user");
        assert_eq!(Role::Assistant.to_string(), "assistant");
    }

    #[test]
    fn to_openai_messages_shape() {
        let mut c = Conversation::new();
        c.push_system("sys".into());
        c.push_user("hi".into());
        c.push_assistant("hello".into());
        let msgs = c.to_openai_messages();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "sys");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[2]["role"], "assistant");
        // No stray fields — OpenAI rejects unknown keys on some models.
        let obj = msgs[0].as_object().unwrap();
        assert_eq!(obj.len(), 2);
    }

    #[test]
    fn push_helpers_record_role() {
        let mut c = Conversation::new();
        c.push_user("q".into());
        c.push_assistant("a".into());
        assert_eq!(c.turns[0].role, Role::User);
        assert_eq!(c.turns[1].role, Role::Assistant);
    }
}
