# halo-agents

17-specialist async registry. `Name` enum (17 variants) + `Specialist` trait
(dyn-safe, JSON I/O, optional description + input_schema + output_schema) +
`TypedSpecialist<I, O>` (AFIT, serde + schemars derives) + `Typed<T>` adapter
that erases a typed specialist into the dyn form.

Part of the strix-ai-rs ecosystem. Consumed by `halo-mcp` to surface
specialists as MCP tools with real JSON Schemas.

## Usage

```rust
use halo_agents::{Registry, Name, TypedSpecialist, Typed};
use serde::{Deserialize, Serialize};
use schemars::JsonSchema;

#[derive(Deserialize, JsonSchema)]
struct MyIn { cmd: String }

#[derive(Serialize, JsonSchema)]
struct MyOut { ok: bool }

struct MySpec;
impl TypedSpecialist for MySpec {
    type Input = MyIn; type Output = MyOut;
    const NAME: Name = Name::Anvil;
    const DESCRIPTION: &'static str = "example";
    async fn handle_typed(&self, req: Self::Input) -> anyhow::Result<Self::Output> {
        Ok(MyOut { ok: req.cmd == "ping" })
    }
}

let mut r = Registry::default_stubs();
r.insert(Typed::arc(MySpec));
let v = r.dispatch("anvil", serde_json::json!({"cmd":"ping"})).await?;
```

## Tests

`cargo test` — 10 green.
