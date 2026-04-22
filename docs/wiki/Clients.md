# Connecting clients

Everything speaks OpenAI-compatible HTTP on `:8180`. Any OpenAI SDK works out of the box. MCP clients attach to `1bit-halo-mcp` on `:8181`.

## Open WebUI

Open WebUI is the blessed third-party client today. Carve-out under Rule A (caller-side only; sunsets on 1bit-helm v0.3 parity).

### Docker path

```bash
docker run -d -p 3000:8080 \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8180/v1 \
  -e OPENAI_API_KEY=none \
  -v openwebui-data:/app/backend/data \
  --name openwebui \
  --restart unless-stopped \
  ghcr.io/open-webui/open-webui:main
```

### Native path (pipx)

```bash
pipx install open-webui
OPENAI_API_BASE_URL=http://127.0.0.1:8180/v1 \
OPENAI_API_KEY=none \
open-webui serve --port 3000
```

Visit `http://localhost:3000`, create the first admin account (stored locally), select **1bit-halo-v2** from the model dropdown.

## Raw HTTP

Handy for smoke tests and shell scripts.

### List models

```bash
curl -s http://127.0.0.1:8180/v1/models | jq '.data[].id'
```

### One-shot completion

```bash
curl -s http://127.0.0.1:8180/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "1bit-halo-v2",
    "messages": [
      {"role":"system","content":"Be concise."},
      {"role":"user","content":"Explain ternary weights in one sentence."}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }' | jq -r '.choices[0].message.content'
```

### Streaming (SSE)

```bash
curl -N http://127.0.0.1:8180/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "1bit-halo-v2",
    "messages": [{"role":"user","content":"count to ten"}],
    "stream": true
  }'
# server-sent events: each chunk is `data: {...}\n\n`
```

## MCP clients

`1bit-halo-mcp` exposes introspection and control as a Model Context Protocol server. Any MCP-aware client (Claude Desktop, Claude Code, Continue, Cursor, custom agents) can attach.

### Claude Desktop

```jsonc
// ~/.config/Claude/claude_desktop_config.json
{
  "mcpServers": {
    "1bit": {
      "command": "/usr/local/bin/1bit-halo-mcp",
      "args": ["--server", "http://127.0.0.1:8180"]
    }
  }
}
```

### Claude Code

```bash
claude mcp add 1bit /usr/local/bin/1bit-halo-mcp -- \
  --server http://127.0.0.1:8180
```

`1bit-halo-mcp` exposes tools for: model listing, health probes, KV-cache stats, active session inspection, sampler overrides (temperature, top-p, top-k), and kernel timing. 22 tests cover the surface as of 2026-04-19.

## Custom / SDK

Any OpenAI SDK works. Examples below in Python (caller-side), TypeScript (caller-side), and Rust.

### Python · openai-python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8180/v1",
    api_key="none",  # 1bit-halo-server ignores the key by default
)

stream = client.chat.completions.create(
    model="1bit-halo-v2",
    messages=[{"role": "user", "content": "Hello, ternary."}],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

### TypeScript · openai (Bun-friendly)

```typescript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8180/v1",
  apiKey: "none",
});

const stream = await client.chat.completions.create({
  model: "1bit-halo-v2",
  messages: [{ role: "user", content: "Hello, ternary." }],
  stream: true,
});

for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content ?? "");
}
```

### Rust · async-openai

```rust
use async_openai::{Client, config::OpenAIConfig, types::*};
use futures::StreamExt;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = OpenAIConfig::new()
        .with_api_base("http://localhost:8180/v1")
        .with_api_key("none");
    let client = Client::with_config(config);

    let req = CreateChatCompletionRequestArgs::default()
        .model("1bit-halo-v2")
        .messages([ChatCompletionRequestUserMessageArgs::default()
            .content("Hello, ternary.")
            .build()?
            .into()])
        .stream(true)
        .build()?;

    let mut stream = client.chat().create_stream(req).await?;
    while let Some(result) = stream.next().await {
        if let Ok(chunk) = result {
            if let Some(content) = &chunk.choices[0].delta.content {
                print!("{content}");
            }
        }
    }
    Ok(())
}
```
