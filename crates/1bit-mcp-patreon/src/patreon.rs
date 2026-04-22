//! Minimal typed subset of the Patreon API v2 surface we consume.
//!
//! We intentionally keep types thin — `serde_json::Value` flows through
//! anywhere the schema is too volatile or too permissive to bake into
//! structs. The MCP surface returns JSON text, so over-modelling buys
//! nothing and makes schema drift a compile error on the wrong side.

use serde::{Deserialize, Serialize};

/// Creator campaign summary. Populated from the `data[]` element of
/// `GET /api/oauth2/v2/campaigns`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Campaign {
    pub id: String,
    #[serde(default)]
    pub creation_name: Option<String>,
    #[serde(default)]
    pub patron_count: Option<u64>,
    #[serde(default)]
    pub is_monthly: Option<bool>,
    #[serde(default)]
    pub published_at: Option<String>,
}

/// Patron tier summary. Drops most rarely-used knobs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tier {
    pub id: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub amount_cents: Option<i64>,
    #[serde(default)]
    pub patron_count: Option<u64>,
    #[serde(default)]
    pub description: Option<String>,
}

/// Member-of-campaign status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PatronStatus {
    ActivePatron,
    DeclinedPatron,
    FormerPatron,
    #[serde(other)]
    Other,
}

/// Member (pledge-holder) summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Member {
    pub id: String,
    #[serde(default)]
    pub email: Option<String>,
    #[serde(default)]
    pub full_name: Option<String>,
    #[serde(default)]
    pub patron_status: Option<PatronStatus>,
    #[serde(default)]
    pub currently_entitled_amount_cents: Option<i64>,
}

/// Post draft body for `POST /api/oauth2/v2/campaigns/{id}/posts`.
#[derive(Debug, Clone, Serialize)]
pub struct PostDraft {
    pub title: String,
    pub content: String,
    /// Patreon's visibility knob — `public_patrons`, `patrons_only`, `public`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub post_type: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patron_status_snake_roundtrips() {
        let s: PatronStatus = serde_json::from_str("\"active_patron\"").unwrap();
        assert_eq!(s, PatronStatus::ActivePatron);
        let s: PatronStatus = serde_json::from_str("\"declined_patron\"").unwrap();
        assert_eq!(s, PatronStatus::DeclinedPatron);
    }

    #[test]
    fn patron_status_unknown_falls_back_to_other() {
        let s: PatronStatus = serde_json::from_str("\"some_future_state\"").unwrap();
        assert_eq!(s, PatronStatus::Other);
    }

    #[test]
    fn campaign_deserializes_with_missing_optionals() {
        let v = serde_json::json!({ "id": "1234" });
        let c: Campaign = serde_json::from_value(v).unwrap();
        assert_eq!(c.id, "1234");
        assert!(c.creation_name.is_none());
        assert!(c.patron_count.is_none());
    }

    #[test]
    fn tier_deserializes_mix_of_fields() {
        let v = serde_json::json!({ "id": "t1", "title": "Supporter", "amount_cents": 500 });
        let t: Tier = serde_json::from_value(v).unwrap();
        assert_eq!(t.id, "t1");
        assert_eq!(t.title.as_deref(), Some("Supporter"));
        assert_eq!(t.amount_cents, Some(500));
    }
}
