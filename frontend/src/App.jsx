import React, { useEffect, useRef, useState } from "react";

import {
  Mic,
  Square,
  ThumbsUp,
  ThumbsDown,
  Volume2,
  MessageSquareText,
} from "lucide-react";

const API = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const WS_API = import.meta.env.VITE_WS_BASE_URL || API.replace(/^http/, "ws");

const BROWSER_LANG = {
  en: "en-IN",
  hi: "hi-IN",
};

// ============================================================
// Voice Activity Detection settings
// ============================================================

const VAD_START_THRESHOLD = Number(
  import.meta.env.VITE_VAD_START_THRESHOLD || 0.018,
);
const VAD_END_THRESHOLD = Number(
  import.meta.env.VITE_VAD_END_THRESHOLD || 0.012,
);
const VAD_SILENCE_MS = Number(import.meta.env.VITE_VAD_SILENCE_MS || 900);
const VAD_MIN_SPEECH_MS = Number(
  import.meta.env.VITE_VAD_MIN_SPEECH_MS || 250,
);
const VAD_MAX_UTTERANCE_MS = Number(
  import.meta.env.VITE_VAD_MAX_UTTERANCE_MS || 20000,
);
const VAD_PREROLL_MS = Number(import.meta.env.VITE_VAD_PREROLL_MS || 250);

// ============================================================
// Helpers
// ============================================================

function base64ToBlobUrl(b64, mime = "audio/mpeg") {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return URL.createObjectURL(new Blob([bytes], { type: mime }));
}

function writeAscii(view, offset, text) {
  for (let i = 0; i < text.length; i++) {
    view.setUint8(offset + i, text.charCodeAt(i));
  }
}

// Each spoken question becomes one standalone WAV file.
function encodeMono16Wav(floatChunks, sampleRate) {
  const totalSamples = floatChunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const dataBytes = totalSamples * 2;
  const buffer = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buffer);

  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, dataBytes, true);

  let offset = 44;
  for (const chunk of floatChunks) {
    for (let i = 0; i < chunk.length; i++) {
      const sample = Math.max(-1, Math.min(1, chunk[i]));
      const pcm16 = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
      view.setInt16(offset, pcm16, true);
      offset += 2;
    }
  }

  return buffer;
}

function browserSpeak(text, language, onDone) {
  if (!("speechSynthesis" in window) || !text) {
    onDone?.();
    return;
  }
  window.speechSynthesis.cancel();
  const utter = new SpeechSynthesisUtterance(text);
  utter.lang = BROWSER_LANG[language] || "en-IN";
  utter.onend = () => onDone?.();
  utter.onerror = () => onDone?.();
  window.speechSynthesis.speak(utter);
}

function voiceStateText(state) {
  switch (state) {
    case "connecting":
      return "Connecting voice session…";
    case "listening":
      return "Listening — start speaking naturally.";
    case "hearing":
      return "Hearing you… pause when you are finished.";
    case "transcribing":
      return "Transcribing speech…";
    case "searching":
      return "Searching documents and generating an answer…";
    case "tts":
      return "Preparing spoken response…";
    case "speaking":
      return "Speaking response…";
    default:
      return "Voice session is off.";
  }
}

// ============================================================
// APP
// ============================================================

export default function App() {
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [language, setLanguage] = useState("en");

  // true -> Document RAG, false -> Direct LLM
  const [ragBased, setRagBased] = useState(true);

  const [textQuery, setTextQuery] = useState("");
  const [correction, setCorrection] = useState("");

  const [voiceState, setVoiceState] = useState("idle");
  const [sessionActive, setSessionActive] = useState(false);
  const [liveTranscript, setLiveTranscript] = useState("");

  // ==========================================================
  // Refs
  // ==========================================================

  const languageRef = useRef(language);
  const ragBasedRef = useRef(ragBased);
  const sessionActiveRef = useRef(false);
  const wsRef = useRef(null);

  const micStreamRef = useRef(null);
  const audioContextRef = useRef(null);
  const sourceNodeRef = useRef(null);
  const processorNodeRef = useRef(null);
  const muteGainRef = useRef(null);

  const preRollRef = useRef([]);
  const preRollSamplesRef = useRef(0);
  const utterancePcmRef = useRef([]);

  const speechActiveRef = useRef(false);
  const silenceStartRef = useRef(null);
  const speechStartedAtRef = useRef(null);
  const speechEnergyMsRef = useRef(0);

  const processingRef = useRef(false);
  const assistantSpeakingRef = useRef(false);

  const currentAudioRef = useRef(null);
  const currentAudioUrlRef = useRef(null);

  useEffect(() => {
    languageRef.current = language;
  }, [language]);

  useEffect(() => {
    ragBasedRef.current = ragBased;
  }, [ragBased]);

  useEffect(() => {
    return () => {
      cleanupVoiceResources(true);
    };
  }, []);

  function resetVadBuffers() {
    preRollRef.current = [];
    preRollSamplesRef.current = 0;
    utterancePcmRef.current = [];
    speechActiveRef.current = false;
    silenceStartRef.current = null;
    speechStartedAtRef.current = null;
    speechEnergyMsRef.current = 0;
  }

  function stopCurrentPlayback() {
    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current.src = "";
      currentAudioRef.current = null;
    }
    if (currentAudioUrlRef.current) {
      URL.revokeObjectURL(currentAudioUrlRef.current);
      currentAudioUrlRef.current = null;
    }
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
  }

  function cleanupVoiceResources(closeSocket = true) {
    sessionActiveRef.current = false;
    processingRef.current = false;
    assistantSpeakingRef.current = false;

    stopCurrentPlayback();
    resetVadBuffers();

    if (processorNodeRef.current) {
      processorNodeRef.current.onaudioprocess = null;
      try {
        processorNodeRef.current.disconnect();
      } catch {}
      processorNodeRef.current = null;
    }

    if (sourceNodeRef.current) {
      try {
        sourceNodeRef.current.disconnect();
      } catch {}
      sourceNodeRef.current = null;
    }

    if (muteGainRef.current) {
      try {
        muteGainRef.current.disconnect();
      } catch {}
      muteGainRef.current = null;
    }

    if (audioContextRef.current) {
      const ctx = audioContextRef.current;
      audioContextRef.current = null;
      if (ctx.state !== "closed") {
        ctx.close().catch(() => {});
      }
    }

    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach((track) => track.stop());
      micStreamRef.current = null;
    }

    if (closeSocket && wsRef.current) {
      const ws = wsRef.current;
      wsRef.current = null;
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close(1000, "voice session ended");
      }
    }
  }

  // Mic listening automatically resumes once the assistant finishes speaking.
  function finishAssistantTurn() {
    stopCurrentPlayback();
    assistantSpeakingRef.current = false;
    processingRef.current = false;
    setBusy(false);
    resetVadBuffers();
    setVoiceState(sessionActiveRef.current ? "listening" : "idle");
  }

  // Sequential playback queue for streamed sentence-level TTS chunks.
  const chunkQueueRef = useRef([]);
  const chunkPlayingRef = useRef(false);
  const turnFinalizedRef = useRef(false);
  const chunksPlayedRef = useRef(0);
  const lastChunksRef = useRef([]); // kept for "speak again" replay
  const pendingResultRef = useRef(null);

  function playNextChunk() {
    const chunk = chunkQueueRef.current.shift();

    if (!chunk) {
      chunkPlayingRef.current = false;
      if (turnFinalizedRef.current) {
        finalizeStreamedTurn();
      }
      return;
    }

    chunkPlayingRef.current = true;
    assistantSpeakingRef.current = true;
    setVoiceState("speaking");
    chunksPlayedRef.current += 1;

    if (!chunk.audio_base64) {
      playNextChunk();
      return;
    }

    const url = base64ToBlobUrl(chunk.audio_base64, chunk.audio_mime || "audio/mpeg");
    currentAudioUrlRef.current = url;
    const audio = new Audio(url);
    currentAudioRef.current = audio;

    let advanced = false;
    const advance = () => {
      if (advanced) {
        return;
      }
      advanced = true;
      URL.revokeObjectURL(url);
      playNextChunk();
    };

    audio.onended = advance;
    audio.onerror = advance;
    audio.play().catch(advance);
  }

  // Called once the last queued chunk has finished playing AND the
  // backend has sent turn.result (order between the two can vary).
  function finalizeStreamedTurn() {
    const data = pendingResultRef.current;
    pendingResultRef.current = null;

    if (data && data.browser_tts_fallback && chunksPlayedRef.current === 0) {
      browserSpeak(data.answer, languageRef.current, finishAssistantTurn);
      return;
    }

    finishAssistantTurn();
  }

  function enqueueTtsChunk(chunk) {
    lastChunksRef.current.push(chunk);
    chunkQueueRef.current.push(chunk);
    if (!chunkPlayingRef.current) {
      playNextChunk();
    }
  }

  function playAssistantResult(data) {
    assistantSpeakingRef.current = true;
    setVoiceState("speaking");

    // Multiple audio events (ended/error/play-rejection) can fire for one
    // playback; only the first should finish the turn or speak a fallback.
    let settled = false;
    const done = () => {
      if (settled) {
        return;
      }
      settled = true;
      finishAssistantTurn();
    };

    if (data.audio_base64) {
      const url = base64ToBlobUrl(data.audio_base64, data.audio_mime || "audio/mpeg");
      currentAudioUrlRef.current = url;
      const audio = new Audio(url);
      currentAudioRef.current = audio;

      // play() can reject even after playback actually starts, so only
      // fall back to browser TTS if audio never started (avoids double voice).
      let started = false;
      audio.onplaying = () => {
        started = true;
      };
      audio.onended = done;
      audio.onerror = () => {
        if (settled) {
          return;
        }
        stopCurrentPlayback();
        browserSpeak(data.answer, languageRef.current, done);
      };
      audio.play().catch(() => {
        if (started || settled) {
          return;
        }
        stopCurrentPlayback();
        browserSpeak(data.answer, languageRef.current, done);
      });
      return;
    }

    if (data.browser_tts_fallback) {
      browserSpeak(data.answer, languageRef.current, done);
      return;
    }

    done();
  }

  function handleWsMessage(event) {
    let message;
    try {
      message = JSON.parse(event.data);
    } catch {
      return;
    }

    switch (message.type) {
      case "session.ready":
        if (sessionActiveRef.current && !processingRef.current) {
          setVoiceState("listening");
        }
        break;

      case "stt.started":
        setVoiceState("transcribing");
        break;

      case "transcript":
        setLiveTranscript(message.text || "");
        setResult({ transcript: message.text || "", answer: "" });
        break;

      case "rag.started":
        setVoiceState("searching");
        break;

      case "answer.delta":
        setResult((previous) => ({
          ...(previous || {}),
          answer: (previous?.answer || "") + (message.text || ""),
        }));
        break;

      case "tts.chunk":
        enqueueTtsChunk({
          audio_base64: message.audio_base64,
          audio_mime: message.audio_mime,
        });
        break;

      case "tts.started":
        setVoiceState("tts");
        break;

      case "turn.result": {
        const data = message.data;
        setResult(data);
        setLiveTranscript(data.transcript || "");
        setCorrection("");

        if (data.streamed) {
          // Audio already arrived as "tts.chunk" events; just wait for
          // the queue to drain, then finish the turn.
          turnFinalizedRef.current = true;
          pendingResultRef.current = data;
          if (!chunkPlayingRef.current) {
            finalizeStreamedTurn();
          }
        } else {
          playAssistantResult(data);
        }
        break;
      }

      case "error":
        setError(message.message || "Voice session error");
        processingRef.current = false;
        assistantSpeakingRef.current = false;
        setBusy(false);
        resetVadBuffers();
        if (sessionActiveRef.current) {
          setVoiceState("listening");
        }
        break;

      default:
        break;
    }
  }

  function connectVoiceSocket() {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(`${WS_API}/api/v1/ws/voice`);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onopen = () => resolve(ws);
      ws.onmessage = handleWsMessage;
      ws.onerror = () => reject(new Error("Could not connect to the voice WebSocket"));

      ws.onclose = (event) => {
        wsRef.current = null;
        if (sessionActiveRef.current && event.code !== 1000) {
          setError(`Voice WebSocket closed unexpectedly (${event.code}).`);
          cleanupVoiceResources(false);
          setSessionActive(false);
          setVoiceState("idle");
          setBusy(false);
        }
      };
    });
  }

  // Keeps ~250ms of audio before speech is detected, so the first syllable isn't clipped.
  function pushPreRoll(frame, sampleRate) {
    preRollRef.current.push(frame);
    preRollSamplesRef.current += frame.length;

    const maxSamples = Math.max(1, Math.floor((sampleRate * VAD_PREROLL_MS) / 1000));
    while (preRollSamplesRef.current > maxSamples && preRollRef.current.length > 1) {
      const removed = preRollRef.current.shift();
      preRollSamplesRef.current -= removed.length;
    }
  }

  function sendCurrentUtterance(sampleRate) {
    if (!speechActiveRef.current) {
      return;
    }

    const pcm = utterancePcmRef.current;
    const speechMs = speechEnergyMsRef.current;

    speechActiveRef.current = false;
    utterancePcmRef.current = [];
    silenceStartRef.current = null;
    speechStartedAtRef.current = null;
    speechEnergyMsRef.current = 0;
    preRollRef.current = [];
    preRollSamplesRef.current = 0;

    // Ignore accidental tiny noises.
    if (speechMs < VAD_MIN_SPEECH_MS || pcm.length === 0) {
      if (sessionActiveRef.current) {
        setVoiceState("listening");
      }
      return;
    }

    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      setError("Voice WebSocket is not connected.");
      if (sessionActiveRef.current) {
        setVoiceState("listening");
      }
      return;
    }

    try {
      const wav = encodeMono16Wav(pcm, sampleRate);

      processingRef.current = true;
      setBusy(true);
      setVoiceState("transcribing");
      setLiveTranscript("");
      setError("");

      // Fresh turn: clear any streaming/playback state left from the previous one.
      chunkQueueRef.current = [];
      chunkPlayingRef.current = false;
      turnFinalizedRef.current = false;
      chunksPlayedRef.current = 0;
      lastChunksRef.current = [];
      pendingResultRef.current = null;

      ws.send(
        JSON.stringify({
          type: "utterance.start",
          language: languageRef.current,
          top_k: 5,
          rag_based: ragBasedRef.current,
          mime_type: "audio/wav",
        }),
      );

      ws.send(wav);
    } catch (e) {
      processingRef.current = false;
      setBusy(false);
      setError(`Could not send voice utterance: ${e}`);
      if (sessionActiveRef.current) {
        setVoiceState("listening");
      }
    }
  }

  async function setupVoiceCapture(stream) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) {
      throw new Error("Web Audio API is not supported in this browser.");
    }

    const ctx = new AudioContextClass();
    audioContextRef.current = ctx;
    await ctx.resume();

    const source = ctx.createMediaStreamSource(stream);
    sourceNodeRef.current = source;

    // ScriptProcessor keeps this self-contained; consider AudioWorklet for production.
    const processor = ctx.createScriptProcessor(4096, 1, 1);
    processorNodeRef.current = processor;

    const mute = ctx.createGain();
    mute.gain.value = 0;
    muteGainRef.current = mute;

    source.connect(processor);
    processor.connect(mute);
    mute.connect(ctx.destination);

    processor.onaudioprocess = (event) => {
      // Ignore mic input while backend is processing or assistant is speaking.
      if (!sessionActiveRef.current || processingRef.current || assistantSpeakingRef.current) {
        return;
      }

      const input = event.inputBuffer.getChannelData(0);
      const frame = new Float32Array(input);
      const sampleRate = event.inputBuffer.sampleRate || ctx.sampleRate;
      const frameMs = (frame.length / sampleRate) * 1000;

      let sumSquares = 0;
      for (let i = 0; i < frame.length; i++) {
        sumSquares += frame[i] * frame[i];
      }
      const rms = Math.sqrt(sumSquares / frame.length);
      const now = performance.now();

      if (!speechActiveRef.current) {
        pushPreRoll(frame, sampleRate);

        if (rms >= VAD_START_THRESHOLD) {
          speechActiveRef.current = true;
          speechStartedAtRef.current = now;
          speechEnergyMsRef.current = frameMs;
          silenceStartRef.current = null;
          utterancePcmRef.current = [...preRollRef.current];
          preRollRef.current = [];
          preRollSamplesRef.current = 0;
          setVoiceState("hearing");
        }
        return;
      }

      utterancePcmRef.current.push(frame);

      if (rms >= VAD_END_THRESHOLD) {
        speechEnergyMsRef.current += frameMs;
        silenceStartRef.current = null;
      } else if (silenceStartRef.current == null) {
        silenceStartRef.current = now;
      }

      const utteranceMs = now - (speechStartedAtRef.current || now);
      const silentForMs = silenceStartRef.current == null ? 0 : now - silenceStartRef.current;

      if (silentForMs >= VAD_SILENCE_MS || utteranceMs >= VAD_MAX_UTTERANCE_MS) {
        sendCurrentUtterance(sampleRate);
      }
    };
  }

  async function startVoiceSession() {
    if (sessionActiveRef.current) {
      return;
    }

    setError("");
    setCorrection("");
    setLiveTranscript("");
    setVoiceState("connecting");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1,
        },
      });

      micStreamRef.current = stream;
      sessionActiveRef.current = true;
      setSessionActive(true);

      await connectVoiceSocket();
      await setupVoiceCapture(stream);

      resetVadBuffers();
      setVoiceState("listening");
    } catch (e) {
      setError(`Could not start voice session: ${e}`);
      cleanupVoiceResources(true);
      setSessionActive(false);
      setVoiceState("idle");
      setBusy(false);
    }
  }

  function endVoiceSession() {
    cleanupVoiceResources(true);
    setSessionActive(false);
    setVoiceState("idle");
    setBusy(false);
    setLiveTranscript("");
  }

  // ==========================================================
  // Text query (unchanged REST path)
  // ==========================================================

  async function submitText() {
    const query = textQuery.trim();
    if (!query) {
      return;
    }

    setBusy(true);
    setError("");
    setResult(null);
    setCorrection("");

    try {
      const r = await fetch(`${API}/api/v1/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, language, top_k: 5, rag_based: ragBased }),
      });

      if (!r.ok) {
        throw new Error(await r.text());
      }

      const data = await r.json();
      setResult({ ...data, transcript: query });
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  function speakAgain() {
    if (!result?.answer) {
      return;
    }

    // Gate microphone during replay.
    processingRef.current = true;
    assistantSpeakingRef.current = true;
    setVoiceState("speaking");

    if (result.streamed && lastChunksRef.current.length > 0) {
      chunkQueueRef.current = [...lastChunksRef.current];
      chunkPlayingRef.current = false;
      turnFinalizedRef.current = true;
      chunksPlayedRef.current = 0;
      pendingResultRef.current = result;
      playNextChunk();
      return;
    }

    playAssistantResult(result);
  }

  async function feedback(rating) {
    if (!result?.request_id) {
      return;
    }

    const r = await fetch(`${API}/api/v1/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        request_id: result.request_id,
        language,
        rating,
        query: result.transcript || textQuery,
        model_answer: result.answer,
        corrected_answer: correction.trim() || null,
        sources: result.sources || [],
        rag_based: result.rag_based ?? ragBased,
      }),
    });

    if (!r.ok) {
      setError(`Feedback failed: ${await r.text()}`);
    }
  }

  // ==========================================================
  // UI
  // ==========================================================

  return (
    <main className="page">
      <section className="card">
        <div className="eyebrow">VOICE + RAG</div>
        <h1>English + Hindi document assistant</h1>
        <p className="sub">
          Ask by voice or text. The app retrieves your English/Hindi documents,
          generates a grounded answer, and can speak the answer back.
        </p>

        {/* LANGUAGE */}
        <div className="language-row">
          <label htmlFor="language">Answer / speech language</label>
          <select id="language" value={language} onChange={(e) => setLanguage(e.target.value)}>
            <option value="en">English</option>
            <option value="hi">Hindi / हिन्दी</option>
          </select>
        </div>

        {/* ANSWER MODE */}
        <div className="language-row">
          <label htmlFor="answer-mode">Answer mode</label>
          <select
            id="answer-mode"
            value={ragBased ? "rag" : "direct"}
            onChange={(e) => setRagBased(e.target.value === "rag")}
          >
            <option value="rag">Document RAG</option>
            <option value="direct">Direct LLM</option>
          </select>
        </div>

        {/* CONTINUOUS VOICE MODE */}
        <div className="mode-block">
          <div className="label">Voice conversation</div>

          <div className="controls">
            {!sessionActive ? (
              <button className="primary" onClick={startVoiceSession}>
                <Mic size={18} />
                Start conversation
              </button>
            ) : (
              <button className="danger" onClick={endVoiceSession}>
                <Square size={18} />
                End conversation
              </button>
            )}
          </div>

          <div className="ready">{voiceStateText(voiceState)}</div>

          {sessionActive && (
            <div className="ready">
              Speak normally. A pause of about {VAD_SILENCE_MS / 1000}s
              automatically sends the current question. The microphone resumes
              after the assistant finishes speaking.
            </div>
          )}

          {liveTranscript && sessionActive && (
            <div className="transcript">Heard: {liveTranscript}</div>
          )}
        </div>

        {/* TEXT QUERY */}
        <div className="mode-block">
          <div className="label">Text query</div>
          <div className="text-query-row">
            <input
              value={textQuery}
              onChange={(e) => setTextQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  submitText();
                }
              }}
              placeholder={language === "hi" ? "अपना सवाल लिखें…" : "Type your question…"}
            />
            <button onClick={submitText} disabled={!textQuery.trim() || busy}>
              <MessageSquareText size={18} />
              Ask text
            </button>
          </div>
        </div>

        {error && <div className="error">{error}</div>}

        {result?.answer && (
          <div className="result">
            <div className="ready">Mode: {result.rag_based ? "Document RAG" : "Direct LLM"}</div>

            <div className="label">Question / transcript</div>
            <div className="transcript">{result.transcript}</div>

            <div className="label row">
              <span>{result.rag_based ? "Grounded answer" : "Direct LLM answer"}</span>
              <button className="icon" onClick={speakAgain}>
                <Volume2 size={18} />
              </button>
            </div>

            <div className="answer">{result.answer}</div>

            <div className="feedback-block">
              <label>Optional human correction for future training</label>
              <textarea
                value={correction}
                onChange={(e) => setCorrection(e.target.value)}
                placeholder="If the answer is wrong, type the correct answer before pressing thumbs down."
              />
              <div className="feedback">
                <span>Was this grounded and useful?</span>
                <button onClick={() => feedback(1)}>
                  <ThumbsUp size={16} />
                </button>
                <button onClick={() => feedback(-1)}>
                  <ThumbsDown size={16} />
                </button>
              </div>
            </div>

            <details>
              <summary>Sources & latency</summary>
              <pre>
                {JSON.stringify(
                  {
                    rag_based: result.rag_based,
                    sources: result.sources,
                    timings_ms: result.timings_ms,
                  },
                  null,
                  2,
                )}
              </pre>
            </details>
          </div>
        )}
      </section>
    </main>
  );
}

