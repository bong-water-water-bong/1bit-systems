// voice/app.js — mouth-to-ear loop on your 1bit.systems box.
// Same-origin API. Caddy routes /whisper, /v2, /kokoro to local services.
(() => {
  const $ = (id) => document.getElementById(id);
  const splash = $('splash');
  const app = $('app');
  const status = $('status');
  const meter = $('meter').firstElementChild;
  const mic = $('mic');
  const transcript = $('transcript');
  const voiceSel = $('voice-sel');
  const modelSel = $('model-sel');

  let mediaStream = null;
  let recorder = null;
  let chunks = [];
  let analyser = null;
  let rafId = 0;
  let busy = false;
  const history = [
    { role: 'system', content: 'You are halo, a helpful local voice assistant. Keep replies under two sentences when the user is talking by voice.' },
  ];

  function setStatus(s) { status.textContent = s; }
  function addMsg(role, text) {
    const p = document.placeholder ? null : null;
    const ph = transcript.querySelector('.voice-placeholder');
    if (ph) ph.remove();
    const el = document.createElement('p');
    el.className = role === 'user' ? 'msg-user' : 'msg-asst';
    el.textContent = text;
    transcript.appendChild(el);
    transcript.scrollTop = transcript.scrollHeight;
  }

  async function ensureMic() {
    if (mediaStream) return mediaStream;
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 }
    });
    const ac = new (window.AudioContext || window.webkitAudioContext)();
    const src = ac.createMediaStreamSource(mediaStream);
    analyser = ac.createAnalyser();
    analyser.fftSize = 512;
    src.connect(analyser);
    pumpMeter();
    return mediaStream;
  }

  function pumpMeter() {
    const buf = new Uint8Array(analyser.frequencyBinCount);
    const tick = () => {
      analyser.getByteTimeDomainData(buf);
      let peak = 0;
      for (let i = 0; i < buf.length; i++) {
        const v = Math.abs(buf[i] - 128);
        if (v > peak) peak = v;
      }
      meter.style.width = Math.min(100, (peak / 128) * 180).toFixed(1) + '%';
      rafId = requestAnimationFrame(tick);
    };
    tick();
  }

  async function start() {
    if (busy) return;
    await ensureMic();
    chunks = [];
    const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus'
      : MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : '';
    recorder = new MediaRecorder(mediaStream, mime ? { mimeType: mime } : undefined);
    recorder.ondataavailable = (e) => { if (e.data && e.data.size > 0) chunks.push(e.data); };
    recorder.onstop = onStop;
    recorder.start();
    mic.classList.add('rec');
    setStatus('listening…');
  }

  async function stop(cancel = false) {
    if (!recorder || recorder.state === 'inactive') return;
    recorder._cancel = cancel;
    recorder.stop();
    mic.classList.remove('rec');
  }

  async function onStop() {
    const cancelled = recorder._cancel;
    const blob = new Blob(chunks, { type: recorder.mimeType || 'audio/webm' });
    recorder = null;
    if (cancelled || blob.size < 2000) { setStatus('idle'); return; }
    busy = true;
    try {
      setStatus('transcribing…');
      const text = await transcribe(blob);
      if (!text) { setStatus('empty'); return; }
      addMsg('user', text);
      history.push({ role: 'user', content: text });

      setStatus('thinking…');
      const reply = await chat();
      addMsg('assistant', reply);
      history.push({ role: 'assistant', content: reply });

      setStatus('speaking…');
      await speak(reply, voiceSel.value);
      setStatus('idle');
    } catch (e) {
      console.error(e);
      setStatus('error: ' + (e.message || e));
    } finally {
      busy = false;
    }
  }

  async function transcribe(blob) {
    const fd = new FormData();
    fd.append('file', blob, 'clip.webm');
    fd.append('model', 'whisper-1');
    const r = await fetch('/whisper/v1/audio/transcriptions', { method: 'POST', body: fd });
    if (!r.ok) throw new Error('whisper ' + r.status);
    const j = await r.json();
    return (j.text || '').trim();
  }

  async function chat() {
    const body = {
      model: modelSel.value,
      messages: history,
      stream: false,
      max_tokens: 256,
      temperature: 0.7,
    };
    const r = await fetch('/v2/chat/completions', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error('chat ' + r.status);
    const j = await r.json();
    return j.choices?.[0]?.message?.content?.trim() || '';
  }

  async function speak(text, voice) {
    const r = await fetch('/kokoro/v1/audio/speech', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model: 'kokoro', input: text, voice, format: 'wav' }),
    });
    if (!r.ok) throw new Error('kokoro ' + r.status);
    const buf = await r.arrayBuffer();
    const url = URL.createObjectURL(new Blob([buf], { type: 'audio/wav' }));
    const audio = new Audio(url);
    await audio.play().catch(() => {});
    await new Promise((res) => {
      audio.onended = res;
      audio.onerror = res;
    });
    URL.revokeObjectURL(url);
  }

  // press-and-hold UX — also works on mouse.
  const press = (e) => { e.preventDefault(); start(); };
  const release = (e) => { e.preventDefault(); stop(false); };
  mic.addEventListener('pointerdown', press);
  mic.addEventListener('pointerup', release);
  mic.addEventListener('pointerleave', (e) => { if (recorder) stop(false); });
  mic.addEventListener('contextmenu', (e) => e.preventDefault());

  $('splash-go').addEventListener('click', async () => {
    try {
      await ensureMic();
    } catch (e) {
      setStatus('mic blocked: ' + e.message);
    }
    splash.hidden = true;
    app.hidden = false;
  });
})();
