// halo-voice mobile web UI — voice-first, text optional.
//
// Default flow:
//   hold-to-speak mic → MediaRecorder webm/opus → POST /whisper/stt
//   → text deltas to /v2/v1/chat/completions (SSE stream=true)
//   → sentence-boundary chunks to /kokoro/tts → Audio.play
//
// Backend routes are Caddy-proxied over the mesh, bearer-gated. Token
// is stored in localStorage once the user has been invited.
//
// No build step. Vanilla ES module. Mobile Safari + Chrome + Firefox.

const CFG = {
  chat_url:    '/v2/v1/chat/completions',
  whisper_url: '/whisper/inference',
  tts_url:     '/kokoro/tts',
  model:       'halo-1bit-2b',
  voice:       'af_sky',  // overridden by picker
  max_tokens:  256,
  temperature: 0.7,
  system_prompt: 'You are Halo, a helpful local AI assistant.',
  assistant_name: 'Halo',
};

// Kokoro voice codes: <accent><gender>_<name>.  We map user picks → voice.
const VOICE_MAP = {
  'a_f': 'af_sky',      // American female
  'a_m': 'am_michael',  // American male
  'a_n': 'af_heart',    // warm neutral-ish
  'b_f': 'bf_emma',     // British female
  'b_m': 'bm_george',   // British male
  'b_n': 'bf_emma',
};

const PERSONALITY_PROMPTS = {
  warm:      'You are warm, conversational, and encouraging. Respond like a friend who genuinely wants to help.',
  formal:    'You are formal, precise, and measured. Avoid slang. Respond as a professional consultant would.',
  playful:   'You are playful and witty. Light jokes, puns, and clever turns of phrase are welcome when they fit.',
  concise:   'Respond in one or two short sentences. No filler, no preamble. Terse and direct.',
  professor: 'You are a patient teacher. Explain ideas from first principles. Use analogies and build up to the answer.',
};

const els = {
  status: document.getElementById('status'),
  log:    document.getElementById('log'),
  mic:    document.getElementById('mic'),
  stop:   document.getElementById('stop'),
  modeBtn:document.getElementById('text-toggle'),
  form:   document.getElementById('text-form'),
  input:  document.getElementById('text-input'),
  back:   document.getElementById('text-back'),
  picker: document.getElementById('picker'),
};

// ─── assistant picker (first visit or when settings reset) ───────────

function loadAssistantSettings() {
  const raw = localStorage.getItem('halo_assistant');
  if (!raw) return null;
  try { return JSON.parse(raw); } catch { return null; }
}

function applyAssistantSettings(s) {
  const voice = VOICE_MAP[`${s.accent}_${s.gender}`] || CFG.voice;
  CFG.voice = voice;
  CFG.assistant_name = s.name || 'Halo';
  const personalityLine = PERSONALITY_PROMPTS[s.personality] || PERSONALITY_PROMPTS.warm;
  CFG.system_prompt =
    `You are ${CFG.assistant_name}, a helpful local AI assistant running on a Strix Halo mini-PC. ${personalityLine}`;
}

function openPicker() {
  if (!els.picker) return;
  els.picker.showModal();
  els.picker.addEventListener('close', () => {
    const data = new FormData(els.picker.querySelector('form'));
    const s = {
      gender:     data.get('gender') || 'f',
      accent:     data.get('accent') || 'a',
      personality:data.get('personality') || 'warm',
      name:       (data.get('assistant_name') || 'Halo').toString().trim() || 'Halo',
    };
    localStorage.setItem('halo_assistant', JSON.stringify(s));
    applyAssistantSettings(s);
    bubble('bot', `Hi, I'm ${CFG.assistant_name}. Tap the mic to talk.`);
    synthAndPlay(`Hi, I'm ${CFG.assistant_name}. Tap the mic to talk.`);
  }, { once: true });
}

// ─── splash screen (every session, dismisses to picker on first visit) ───

function showSplash(onDismiss) {
  const el = document.getElementById('splash');
  if (!el) { onDismiss(); return; }

  const reduced = window.matchMedia &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  let dismissed = false;
  let autoTimer = null;

  const finish = () => {
    if (dismissed) return;
    dismissed = true;
    if (autoTimer) clearTimeout(autoTimer);
    el.classList.add('dismissing');
    const cleanupDelay = reduced ? 150 : 400;
    setTimeout(() => {
      el.remove();
      try { onDismiss(); } catch (e) { /* ignore */ }
    }, cleanupDelay);
  };

  const enterBtn = el.querySelector('.enter');
  if (enterBtn) {
    enterBtn.addEventListener('click', (e) => { e.stopPropagation(); finish(); });
    // First-focusable: focus the Enter button for keyboard users.
    try { enterBtn.focus({ preventScroll: true }); } catch { enterBtn.focus(); }
  }
  el.addEventListener('click', finish);
  const escHandler = (e) => {
    if (e.key === 'Escape') { e.preventDefault(); finish(); }
  };
  document.addEventListener('keydown', escHandler);
  // Clean up the escape listener once finished.
  const origFinish = finish;
  // wrap via closure: remove listener on dismissal
  el.addEventListener('transitionend', () => {
    document.removeEventListener('keydown', escHandler);
  }, { once: true });

  // Auto-dismiss after 5s unless reduced-motion is requested.
  if (!reduced) {
    autoTimer = setTimeout(finish, 5000);
  }
}

(function initAssistant() {
  const existing = loadAssistantSettings();

  const afterSplash = () => {
    if (existing) {
      applyAssistantSettings(existing);
    } else {
      openPicker();
    }
  };

  const start = () => {
    if (existing) {
      // Apply settings immediately so pings/etc work; splash just visual.
      applyAssistantSettings(existing);
      showSplash(() => { /* settings already applied */ });
    } else {
      showSplash(afterSplash);
    }
  };

  if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', start, { once: true });
  } else {
    start();
  }
})();

let bearer = localStorage.getItem('halo_bearer') || '';
if (!bearer) {
  const prompted = prompt('Paste your halo-ai bearer token (sk-halo-…):');
  if (prompted) {
    bearer = prompted.trim();
    localStorage.setItem('halo_bearer', bearer);
  }
}

const authHeaders = (extra = {}) => ({
  'Authorization': 'Bearer ' + bearer,
  ...extra,
});

function setStatus(text, cls) {
  els.status.textContent = text;
  els.status.className = 'status ' + (cls || '');
}

function bubble(kind, text = '') {
  const b = document.createElement('div');
  b.className = 'bubble ' + kind;
  b.textContent = text;
  els.log.appendChild(b);
  b.scrollIntoView({ behavior: 'smooth', block: 'end' });
  return b;
}

// Ping the chat endpoint once on load. Green dot if bearer is valid.
(async function ping() {
  try {
    const r = await fetch('/v2/v1/models', { headers: authHeaders() });
    if (r.ok) setStatus('online', 'ok');
    else      setStatus('auth ' + r.status, 'err');
  } catch (e) {
    setStatus('offline', 'err');
  }
})();

// ─── recording ───────────────────────────────────────────────────────

let rec = null;
let chunks = [];
let liveBubble = null;

async function startRecord() {
  if (rec) return;
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    setStatus('mic denied', 'err');
    return;
  }
  chunks = [];
  rec = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
  rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  rec.onstop = () => {
    stream.getTracks().forEach(t => t.stop());
    const blob = new Blob(chunks, { type: 'audio/webm' });
    chunks = [];
    rec = null;
    handleClip(blob);
  };
  rec.start();
  els.mic.classList.add('live');
  liveBubble = bubble('listening', 'listening…');
  setStatus('listening', 'ok');
}

function stopRecord() {
  if (!rec) return;
  rec.stop();
  els.mic.classList.remove('live');
  setStatus('transcribing', 'ok');
}

// Tap OR press-and-hold.  Short tap = toggle; press-and-hold = while-held.
let holding = false;
let holdTimer = null;

els.mic.addEventListener('pointerdown', () => {
  holding = false;
  holdTimer = setTimeout(() => { holding = true; startRecord(); }, 180);
});
els.mic.addEventListener('pointerup', () => {
  clearTimeout(holdTimer);
  if (holding)       { stopRecord(); }
  else if (rec)      { stopRecord(); }
  else               { startRecord(); }
});
els.mic.addEventListener('pointercancel', () => {
  clearTimeout(holdTimer);
  if (rec) stopRecord();
});

// ─── transcribe + chat + tts ─────────────────────────────────────────

async function handleClip(blob) {
  const form = new FormData();
  form.append('file', blob, 'clip.webm');
  form.append('response_format', 'json');
  form.append('temperature', '0.0');
  let text = '';
  try {
    const r = await fetch(CFG.whisper_url, {
      method: 'POST',
      headers: authHeaders(),
      body: form,
    });
    if (!r.ok) throw new Error('stt ' + r.status);
    const j = await r.json();
    text = (j.text || '').trim();
  } catch (e) {
    setStatus('stt error', 'err');
    if (liveBubble) liveBubble.remove();
    return;
  }
  if (liveBubble) liveBubble.remove();
  if (!text) { setStatus('online', 'ok'); return; }
  bubble('user', text);
  await sendPrompt(text);
}

async function sendPrompt(prompt) {
  setStatus('thinking', 'ok');
  const botBubble = bubble('bot', '');
  const body = {
    model: CFG.model,
    messages: [{ role: 'user', content: prompt }],
    max_tokens: CFG.max_tokens,
    temperature: CFG.temperature,
    stream: true,
  };
  const r = await fetch(CFG.chat_url, {
    method: 'POST',
    headers: authHeaders({ 'content-type': 'application/json' }),
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    botBubble.textContent = 'error: ' + r.status;
    setStatus('error ' + r.status, 'err');
    return;
  }
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  let sentenceBuf = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf('\n\n')) !== -1) {
      const event = buf.slice(0, nl);
      buf = buf.slice(nl + 2);
      const delta = parseSSE(event);
      if (delta) {
        botBubble.textContent += delta;
        sentenceBuf += delta;
        sentenceBuf = flushSentences(sentenceBuf);
      }
    }
  }
  if (sentenceBuf.trim()) synthAndPlay(sentenceBuf.trim());
  setStatus('online', 'ok');
}

function parseSSE(event) {
  for (const line of event.split('\n')) {
    if (!line.startsWith('data:')) continue;
    const payload = line.slice(5).trim();
    if (payload === '[DONE]') return '';
    try {
      const j = JSON.parse(payload);
      return j.choices?.[0]?.delta?.content || '';
    } catch { /* ignore */ }
  }
  return '';
}

function flushSentences(buf) {
  const re = /[.!?]\s+|\n/;
  let m;
  while ((m = buf.match(re))) {
    const idx = m.index + m[0].length;
    const sentence = buf.slice(0, idx).trim();
    buf = buf.slice(idx);
    if (hasSpeakable(sentence)) synthAndPlay(sentence);
  }
  return buf;
}

function hasSpeakable(s) {
  return /[a-zA-Z0-9]/.test(s);
}

async function synthAndPlay(text) {
  try {
    const r = await fetch(CFG.tts_url, {
      method: 'POST',
      headers: authHeaders({ 'content-type': 'application/json' }),
      body: JSON.stringify({ text, voice: CFG.voice }),
    });
    if (!r.ok) return;
    const wavBlob = await r.blob();
    const url = URL.createObjectURL(wavBlob);
    const audio = new Audio(url);
    audio.addEventListener('ended', () => URL.revokeObjectURL(url));
    audio.play().catch(() => { /* autoplay blocked — ignore */ });
  } catch { /* ignore */ }
}

// ─── text mode toggle ────────────────────────────────────────────────

els.modeBtn.addEventListener('click', () => {
  els.form.setAttribute('data-active', '1');
  els.form.hidden = false;
  document.querySelector('footer').hidden = true;
  els.input.focus();
});

els.back.addEventListener('click', () => {
  els.form.removeAttribute('data-active');
  els.form.hidden = true;
  document.querySelector('footer').hidden = false;
});

els.form.addEventListener('submit', (e) => {
  e.preventDefault();
  const text = els.input.value.trim();
  if (!text) return;
  els.input.value = '';
  bubble('user', text);
  sendPrompt(text);
});
