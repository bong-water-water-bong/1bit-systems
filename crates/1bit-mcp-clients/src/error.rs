use thiserror::Error;

#[derive(Debug, Error)]
pub enum McpError {
    #[error("transport i/o: {0}")]
    Io(#[from] std::io::Error),
    #[error("http transport: {0}")]
    Http(#[from] reqwest::Error),
    #[error("json encode/decode: {0}")]
    Json(#[from] serde_json::Error),
    #[error("server returned error code {code}: {message}")]
    Rpc { code: i64, message: String },
    #[error("protocol violation: {0}")]
    Protocol(String),
    #[error("transport closed before response arrived")]
    Closed,
}
