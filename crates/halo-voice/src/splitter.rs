//! Stateful sentence-boundary splitter.
//!
//! Fed one LLM delta at a time (strings of any length, including empty or
//! partial-word), emits complete sentences as they close. A "sentence" is
//! whatever runs up to and includes one of `.`, `!`, `?`, or `\n`.
//!
//! Edge cases the test matrix covers:
//! * Delta ending mid-word — buffered, not emitted.
//! * Multiple sentences in a single delta — emitted in order, buffer ends empty.
//! * Abbreviation false positive (`e.g.`, `Dr.`, `U.S.A.`) — current stance:
//!   we accept false sentence breaks. Voice quality loss is small; the
//!   alternative (carrying enough context to decide) doubles the code.
//!   Documented in `docs/wiki/Why-halo-voice.md`.
//! * Trailing text without a boundary — `finish()` flushes it as a final
//!   partial sentence.
//! * Unicode — boundaries are ASCII bytes in Llama-3 tokenizer output, so
//!   we scan bytes. Works for all UTF-8 inputs because the boundary bytes
//!   (`0x21` / `0x2E` / `0x3F` / `0x0A`) never appear inside a multi-byte
//!   UTF-8 sequence.

/// Character set we treat as end-of-sentence. Order matters for no reason;
/// all four are equally weighted.
pub const BOUNDARY_BYTES: [u8; 4] = [b'.', b'!', b'?', b'\n'];

/// Does the sentence contain any alphanumeric glyph? If not, it's punctuation
/// + whitespace only (lone `"!"`, `"."`, `"..."`) — skip it. TTS would otherwise
///   synthesize a click or silence frame. Live-run edge case 2026-04-20.
fn has_speakable(s: &str) -> bool {
    s.chars().any(|c| c.is_alphanumeric())
}

/// Streaming splitter. Not `Send + Sync` (holds a `String` buffer) but can
/// be wrapped in `Mutex` trivially if a consumer needs cross-task use.
#[derive(Debug, Default)]
pub struct SentenceSplitter {
    buf: String,
}

impl SentenceSplitter {
    pub fn new() -> Self { Self::default() }

    /// Feed a new delta. Returns zero or more complete sentences, in
    /// order. Partial-sentence remainder stays in the internal buffer
    /// until a boundary or `finish()` arrives.
    pub fn feed(&mut self, delta: &str) -> Vec<String> {
        self.buf.push_str(delta);
        let mut out = Vec::new();
        let mut last = 0usize;
        let bytes = self.buf.as_bytes();
        for (i, b) in bytes.iter().enumerate() {
            if BOUNDARY_BYTES.contains(b) {
                let sentence = self.buf[last..=i].trim().to_string();
                if has_speakable(&sentence) { out.push(sentence); }
                last = i + 1;
            }
        }
        if last > 0 {
            self.buf = self.buf[last..].to_string();
        }
        out
    }

    /// Flush any buffered partial sentence. Call once at end-of-stream.
    /// Returns `Some(sentence)` if the buffer was non-empty, else `None`.
    pub fn finish(&mut self) -> Option<String> {
        let trimmed = self.buf.trim();
        if trimmed.is_empty() {
            self.buf.clear();
            None
        } else {
            let out = trimmed.to_string();
            self.buf.clear();
            Some(out)
        }
    }

    /// Peek the current unflushed buffer. Mostly for tests.
    pub fn buffered(&self) -> &str { &self.buf }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_complete_sentence() {
        let mut s = SentenceSplitter::new();
        let out = s.feed("Hello world.");
        assert_eq!(out, vec!["Hello world."]);
        assert_eq!(s.buffered(), "");
        assert_eq!(s.finish(), None);
    }

    #[test]
    fn partial_sentence_buffers() {
        let mut s = SentenceSplitter::new();
        assert!(s.feed("Hello wor").is_empty());
        assert_eq!(s.buffered(), "Hello wor");
        let out = s.feed("ld.");
        assert_eq!(out, vec!["Hello world."]);
    }

    #[test]
    fn multiple_sentences_one_delta() {
        let mut s = SentenceSplitter::new();
        let out = s.feed("One. Two! Three?");
        assert_eq!(out, vec!["One.", "Two!", "Three?"]);
        assert_eq!(s.buffered(), "");
    }

    #[test]
    fn newline_is_a_boundary() {
        let mut s = SentenceSplitter::new();
        let out = s.feed("first line\nsecond line\n");
        assert_eq!(out, vec!["first line", "second line"]);
    }

    #[test]
    fn trailing_partial_flushed_on_finish() {
        let mut s = SentenceSplitter::new();
        assert!(s.feed("trailing no punct").is_empty());
        assert_eq!(s.finish(), Some("trailing no punct".to_string()));
        assert_eq!(s.buffered(), "");
    }

    #[test]
    fn empty_deltas_are_noops() {
        let mut s = SentenceSplitter::new();
        assert!(s.feed("").is_empty());
        assert!(s.feed("").is_empty());
        assert_eq!(s.finish(), None);
    }

    #[test]
    fn unicode_survives() {
        let mut s = SentenceSplitter::new();
        let out = s.feed("café brûlé. émoji 🎭!");
        assert_eq!(out, vec!["café brûlé.", "émoji 🎭!"]);
    }

    #[test]
    fn abbreviation_false_positive_documented() {
        // This is the known false positive — we accept it. If it ever
        // matters, we widen the boundary-decision context (O(1) lookahead
        // on the next char for [A-Z]) rather than add a dictionary.
        let mut s = SentenceSplitter::new();
        let out = s.feed("e.g. this.");
        assert_eq!(out, vec!["e.", "g.", "this."]);
    }

    #[test]
    fn lone_punctuation_delta_is_dropped() {
        // Live-run 2026-04-20: model emitted "Hello" then "!" as separate SSE
        // deltas AFTER an already-closed prior sentence ("Hi."), leaving buf
        // empty when "!" arrived. Old behaviour emitted "!" as its own
        // sentence; TTS would click. Drop it instead.
        let mut s = SentenceSplitter::new();
        assert_eq!(s.feed("Hi."), vec!["Hi."]);
        let out = s.feed("!");
        assert!(out.is_empty(), "lone punctuation must not emit a sentence");
    }

    #[test]
    fn ellipsis_emits_at_most_one_speakable_sentence() {
        let mut s = SentenceSplitter::new();
        let out = s.feed("Hello...");
        assert_eq!(out, vec!["Hello."]);
        // remaining ".." is punctuation-only, no emission on flush
        assert_eq!(s.finish(), None);
    }

    #[test]
    fn whitespace_between_sentences_swallowed() {
        let mut s = SentenceSplitter::new();
        let out = s.feed("First.   Second.");
        assert_eq!(out, vec!["First.", "Second."]);
    }
}
