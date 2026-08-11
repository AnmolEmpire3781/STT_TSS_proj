// // import React, { useEffect, useRef, useState } from "react";
// // import {
// //   Mic,
// //   Square,
// //   ThumbsUp,
// //   ThumbsDown,
// //   Volume2,
// //   MessageSquareText,
// // } from "lucide-react";

// // const API = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

// // const WS_API = import.meta.env.VITE_WS_BASE_URL || API.replace(/^http/, "ws");

// // const BROWSER_LANG = {
// //   en: "en-IN",
// //   hi: "hi-IN",
// // };

// // // ============================================================
// // // Voice Activity Detection settings
// // // ============================================================

// // const VAD_START_THRESHOLD = Number(
// //   import.meta.env.VITE_VAD_START_THRESHOLD || 0.018,
// // );

// // const VAD_END_THRESHOLD = Number(
// //   import.meta.env.VITE_VAD_END_THRESHOLD || 0.012,
// // );

// // const VAD_SILENCE_MS = Number(import.meta.env.VITE_VAD_SILENCE_MS || 900);

// // const VAD_MIN_SPEECH_MS = Number(import.meta.env.VITE_VAD_MIN_SPEECH_MS || 250);

// // const VAD_MAX_UTTERANCE_MS = Number(
// //   import.meta.env.VITE_VAD_MAX_UTTERANCE_MS || 20000,
// // );

// // const VAD_PREROLL_MS = Number(import.meta.env.VITE_VAD_PREROLL_MS || 250);

// // // ============================================================
// // // Helpers
// // // ============================================================

// // function base64ToBlobUrl(b64, mime = "audio/mpeg") {
// //   const binary = atob(b64);

// //   const bytes = new Uint8Array(binary.length);

// //   for (let i = 0; i < binary.length; i++) {
// //     bytes[i] = binary.charCodeAt(i);
// //   }

// //   return URL.createObjectURL(new Blob([bytes], { type: mime }));
// // }

// // function writeAscii(view, offset, text) {
// //   for (let i = 0; i < text.length; i++) {
// //     view.setUint8(offset + i, text.charCodeAt(i));
// //   }
// // }

// // // ============================================================
// // // Convert microphone PCM into a complete WAV file
// // //
// // // Each spoken question becomes one standalone WAV.
// // //
// // // This is better than continuously sending MediaRecorder WebM
// // // chunks because later WebM chunks may depend on the original
// // // container header.
// // // ============================================================

// // function encodeMono16Wav(floatChunks, sampleRate) {
// //   const totalSamples = floatChunks.reduce(
// //     (sum, chunk) => sum + chunk.length,
// //     0,
// //   );

// //   const dataBytes = totalSamples * 2;

// //   const buffer = new ArrayBuffer(44 + dataBytes);

// //   const view = new DataView(buffer);

// //   // RIFF header

// //   writeAscii(view, 0, "RIFF");

// //   view.setUint32(4, 36 + dataBytes, true);

// //   writeAscii(view, 8, "WAVE");

// //   // fmt chunk

// //   writeAscii(view, 12, "fmt ");

// //   view.setUint32(16, 16, true);

// //   // PCM
// //   view.setUint16(20, 1, true);

// //   // mono
// //   view.setUint16(22, 1, true);

// //   view.setUint32(24, sampleRate, true);

// //   view.setUint32(28, sampleRate * 2, true);

// //   view.setUint16(32, 2, true);

// //   view.setUint16(34, 16, true);

// //   // data chunk

// //   writeAscii(view, 36, "data");

// //   view.setUint32(40, dataBytes, true);

// //   let offset = 44;

// //   for (const chunk of floatChunks) {
// //     for (let i = 0; i < chunk.length; i++) {
// //       const sample = Math.max(-1, Math.min(1, chunk[i]));

// //       const pcm16 = sample < 0 ? sample * 0x8000 : sample * 0x7fff;

// //       view.setInt16(offset, pcm16, true);

// //       offset += 2;
// //     }
// //   }

// //   return buffer;
// // }

// // // ============================================================
// // // Browser TTS fallback
// // // ============================================================

// // function browserSpeak(text, language, onDone) {
// //   if (!("speechSynthesis" in window) || !text) {
// //     onDone?.();
// //     return;
// //   }

// //   window.speechSynthesis.cancel();

// //   const utter = new SpeechSynthesisUtterance(text);

// //   utter.lang = BROWSER_LANG[language] || "en-IN";

// //   utter.onend = () => onDone?.();

// //   utter.onerror = () => onDone?.();

// //   window.speechSynthesis.speak(utter);
// // }

// // // ============================================================
// // // UI state message
// // // ============================================================

// // function voiceStateText(state) {
// //   switch (state) {
// //     case "connecting":
// //       return "Connecting voice session…";

// //     case "listening":
// //       return "Listening — start speaking naturally.";

// //     case "hearing":
// //       return "Hearing you… pause when you are finished.";

// //     case "transcribing":
// //       return "Transcribing speech…";

// //     case "searching":
// //       return "Searching documents and generating an answer…";

// //     case "tts":
// //       return "Preparing spoken response…";

// //     case "speaking":
// //       return "Speaking response…";

// //     default:
// //       return "Voice session is off.";
// //   }
// // }

// // // ============================================================
// // // APP
// // // ============================================================

// // export default function App() {
// //   const [busy, setBusy] = useState(false);

// //   const [result, setResult] = useState(null);

// //   const [error, setError] = useState("");

// //   const [language, setLanguage] = useState("en");

// //   const [textQuery, setTextQuery] = useState("");

// //   const [correction, setCorrection] = useState("");

// //   const [voiceState, setVoiceState] = useState("idle");

// //   const [sessionActive, setSessionActive] = useState(false);

// //   const [liveTranscript, setLiveTranscript] = useState("");

// //   // ==========================================================
// //   // Refs
// //   // ==========================================================

// //   const languageRef = useRef(language);

// //   const sessionActiveRef = useRef(false);

// //   const wsRef = useRef(null);

// //   // Microphone / WebAudio

// //   const micStreamRef = useRef(null);

// //   const audioContextRef = useRef(null);

// //   const sourceNodeRef = useRef(null);

// //   const processorNodeRef = useRef(null);

// //   const muteGainRef = useRef(null);

// //   // Voice buffers

// //   const preRollRef = useRef([]);

// //   const preRollSamplesRef = useRef(0);

// //   const utterancePcmRef = useRef([]);

// //   // VAD state

// //   const speechActiveRef = useRef(false);

// //   const silenceStartRef = useRef(null);

// //   const speechStartedAtRef = useRef(null);

// //   const speechEnergyMsRef = useRef(0);

// //   // Turn state

// //   const processingRef = useRef(false);

// //   const assistantSpeakingRef = useRef(false);

// //   // TTS playback

// //   const currentAudioRef = useRef(null);

// //   const currentAudioUrlRef = useRef(null);

// //   // ==========================================================
// //   // Keep latest language available inside event callbacks
// //   // ==========================================================

// //   useEffect(() => {
// //     languageRef.current = language;
// //   }, [language]);

// //   // ==========================================================
// //   // Cleanup on page close
// //   // ==========================================================

// //   useEffect(() => {
// //     return () => {
// //       cleanupVoiceResources(true);
// //     };
// //   }, []);

// //   // ==========================================================
// //   // Reset VAD
// //   // ==========================================================

// //   function resetVadBuffers() {
// //     preRollRef.current = [];

// //     preRollSamplesRef.current = 0;

// //     utterancePcmRef.current = [];

// //     speechActiveRef.current = false;

// //     silenceStartRef.current = null;

// //     speechStartedAtRef.current = null;

// //     speechEnergyMsRef.current = 0;
// //   }

// //   // ==========================================================
// //   // Stop currently playing TTS
// //   // ==========================================================

// //   function stopCurrentPlayback() {
// //     if (currentAudioRef.current) {
// //       currentAudioRef.current.pause();

// //       currentAudioRef.current.src = "";

// //       currentAudioRef.current = null;
// //     }

// //     if (currentAudioUrlRef.current) {
// //       URL.revokeObjectURL(currentAudioUrlRef.current);

// //       currentAudioUrlRef.current = null;
// //     }

// //     if ("speechSynthesis" in window) {
// //       window.speechSynthesis.cancel();
// //     }
// //   }

// //   // ==========================================================
// //   // Cleanup complete voice session
// //   // ==========================================================

// //   function cleanupVoiceResources(closeSocket = true) {
// //     sessionActiveRef.current = false;

// //     processingRef.current = false;

// //     assistantSpeakingRef.current = false;

// //     stopCurrentPlayback();

// //     resetVadBuffers();

// //     if (processorNodeRef.current) {
// //       processorNodeRef.current.onaudioprocess = null;

// //       try {
// //         processorNodeRef.current.disconnect();
// //       } catch {}

// //       processorNodeRef.current = null;
// //     }

// //     if (sourceNodeRef.current) {
// //       try {
// //         sourceNodeRef.current.disconnect();
// //       } catch {}

// //       sourceNodeRef.current = null;
// //     }

// //     if (muteGainRef.current) {
// //       try {
// //         muteGainRef.current.disconnect();
// //       } catch {}

// //       muteGainRef.current = null;
// //     }

// //     if (audioContextRef.current) {
// //       const ctx = audioContextRef.current;

// //       audioContextRef.current = null;

// //       if (ctx.state !== "closed") {
// //         ctx.close().catch(() => {});
// //       }
// //     }

// //     if (micStreamRef.current) {
// //       micStreamRef.current.getTracks().forEach((track) => track.stop());

// //       micStreamRef.current = null;
// //     }

// //     if (closeSocket && wsRef.current) {
// //       const ws = wsRef.current;

// //       wsRef.current = null;

// //       if (
// //         ws.readyState === WebSocket.OPEN ||
// //         ws.readyState === WebSocket.CONNECTING
// //       ) {
// //         ws.close(1000, "voice session ended");
// //       }
// //     }
// //   }

// //   // ==========================================================
// //   // Assistant finished speaking
// //   //
// //   // IMPORTANT:
// //   // microphone listening automatically resumes here.
// //   // ==========================================================

// //   function finishAssistantTurn() {
// //     stopCurrentPlayback();

// //     assistantSpeakingRef.current = false;

// //     processingRef.current = false;

// //     setBusy(false);

// //     resetVadBuffers();

// //     if (sessionActiveRef.current) {
// //       setVoiceState("listening");
// //     } else {
// //       setVoiceState("idle");
// //     }
// //   }

// //   // ==========================================================
// //   // Play assistant TTS
// //   // ==========================================================

// //   function playAssistantResult(data) {
// //     assistantSpeakingRef.current = true;

// //     setVoiceState("speaking");

// //     const done = () => finishAssistantTurn();

// //     if (data.audio_base64) {
// //       const url = base64ToBlobUrl(
// //         data.audio_base64,

// //         data.audio_mime || "audio/mpeg",
// //       );

// //       currentAudioUrlRef.current = url;

// //       const audio = new Audio(url);

// //       currentAudioRef.current = audio;

// //       audio.onended = done;

// //       audio.onerror = () => {
// //         stopCurrentPlayback();

// //         browserSpeak(
// //           data.answer,

// //           languageRef.current,

// //           done,
// //         );
// //       };

// //       audio.play().catch(() => {
// //         stopCurrentPlayback();

// //         browserSpeak(
// //           data.answer,

// //           languageRef.current,

// //           done,
// //         );
// //       });

// //       return;
// //     }

// //     if (data.browser_tts_fallback) {
// //       browserSpeak(
// //         data.answer,

// //         languageRef.current,

// //         done,
// //       );

// //       return;
// //     }

// //     done();
// //   }

// //   // ==========================================================
// //   // WebSocket messages from FastAPI
// //   // ==========================================================

// //   function handleWsMessage(event) {
// //     let message;

// //     try {
// //       message = JSON.parse(event.data);
// //     } catch {
// //       return;
// //     }

// //     switch (message.type) {
// //       case "session.ready":
// //         if (sessionActiveRef.current && !processingRef.current) {
// //           setVoiceState("listening");
// //         }

// //         break;

// //       case "stt.started":
// //         setVoiceState("transcribing");

// //         break;

// //       case "transcript":
// //         setLiveTranscript(message.text || "");

// //         break;

// //       case "rag.started":
// //         setVoiceState("searching");

// //         break;

// //       case "answer":
// //         // Show text before TTS completes.

// //         setResult((previous) => ({
// //           ...(previous || {}),

// //           transcript: message.transcript || "",

// //           answer: message.text || "",

// //           sources: message.sources || [],

// //           timings_ms: message.timings_ms || {},
// //         }));

// //         break;

// //       case "tts.started":
// //         setVoiceState("tts");

// //         break;

// //       case "turn.result": {
// //         const data = message.data;

// //         setResult(data);

// //         setLiveTranscript(data.transcript || "");

// //         setCorrection("");

// //         playAssistantResult(data);

// //         break;
// //       }

// //       case "error":
// //         setError(message.message || "Voice session error");

// //         processingRef.current = false;

// //         assistantSpeakingRef.current = false;

// //         setBusy(false);

// //         resetVadBuffers();

// //         if (sessionActiveRef.current) {
// //           setVoiceState("listening");
// //         }

// //         break;

// //       default:
// //         break;
// //     }
// //   }

// //   // ==========================================================
// //   // Connect FastAPI WebSocket
// //   // ==========================================================

// //   function connectVoiceSocket() {
// //     return new Promise((resolve, reject) => {
// //       const ws = new WebSocket(`${WS_API}/api/v1/ws/voice`);

// //       ws.binaryType = "arraybuffer";

// //       wsRef.current = ws;

// //       ws.onopen = () => {
// //         resolve(ws);
// //       };

// //       ws.onmessage = handleWsMessage;

// //       ws.onerror = () => {
// //         reject(new Error("Could not connect to the voice WebSocket"));
// //       };

// //       ws.onclose = (event) => {
// //         wsRef.current = null;

// //         if (sessionActiveRef.current && event.code !== 1000) {
// //           setError(`Voice WebSocket closed unexpectedly (${event.code}).`);

// //           cleanupVoiceResources(false);

// //           setSessionActive(false);

// //           setVoiceState("idle");

// //           setBusy(false);
// //         }
// //       };
// //     });
// //   }

// //   // ==========================================================
// //   // Maintain ~250 ms audio before VAD speech detection.
// //   //
// //   // This prevents clipping the first syllable.
// //   // ==========================================================

// //   function pushPreRoll(frame, sampleRate) {
// //     preRollRef.current.push(frame);

// //     preRollSamplesRef.current += frame.length;

// //     const maxSamples = Math.max(
// //       1,

// //       Math.floor((sampleRate * VAD_PREROLL_MS) / 1000),
// //     );

// //     while (
// //       preRollSamplesRef.current > maxSamples &&
// //       preRollRef.current.length > 1
// //     ) {
// //       const removed = preRollRef.current.shift();

// //       preRollSamplesRef.current -= removed.length;
// //     }
// //   }

// //   // ==========================================================
// //   // Current utterance ended:
// //   //
// //   // silence OR maximum duration
// //   //      ↓
// //   // create WAV
// //   //      ↓
// //   // send metadata
// //   //      ↓
// //   // send binary WAV
// //   // ==========================================================

// //   function sendCurrentUtterance(sampleRate) {
// //     if (!speechActiveRef.current) {
// //       return;
// //     }

// //     const pcm = utterancePcmRef.current;

// //     const speechMs = speechEnergyMsRef.current;

// //     speechActiveRef.current = false;

// //     utterancePcmRef.current = [];

// //     silenceStartRef.current = null;

// //     speechStartedAtRef.current = null;

// //     speechEnergyMsRef.current = 0;

// //     preRollRef.current = [];

// //     preRollSamplesRef.current = 0;

// //     // Ignore accidental tiny noises.

// //     if (speechMs < VAD_MIN_SPEECH_MS || pcm.length === 0) {
// //       if (sessionActiveRef.current) {
// //         setVoiceState("listening");
// //       }

// //       return;
// //     }

// //     const ws = wsRef.current;

// //     if (!ws || ws.readyState !== WebSocket.OPEN) {
// //       setError("Voice WebSocket is not connected.");

// //       if (sessionActiveRef.current) {
// //         setVoiceState("listening");
// //       }

// //       return;
// //     }

// //     try {
// //       const wav = encodeMono16Wav(
// //         pcm,

// //         sampleRate,
// //       );

// //       processingRef.current = true;

// //       setBusy(true);

// //       setVoiceState("transcribing");

// //       setLiveTranscript("");

// //       setError("");

// //       // ------------------------------------------------------
// //       // Message 1 = JSON metadata
// //       // ------------------------------------------------------

// //       ws.send(
// //         JSON.stringify({
// //           type: "utterance.start",

// //           language: languageRef.current,

// //           top_k: 5,

// //           mime_type: "audio/wav",
// //         }),
// //       );

// //       // ------------------------------------------------------
// //       // Message 2 = actual WAV bytes
// //       // ------------------------------------------------------

// //       ws.send(wav);
// //     } catch (e) {
// //       processingRef.current = false;

// //       setBusy(false);

// //       setError(`Could not send voice utterance: ${e}`);

// //       if (sessionActiveRef.current) {
// //         setVoiceState("listening");
// //       }
// //     }
// //   }

// //   // ==========================================================
// //   // Microphone + VAD
// //   // ==========================================================

// //   async function setupVoiceCapture(stream) {
// //     const AudioContextClass = window.AudioContext || window.webkitAudioContext;

// //     if (!AudioContextClass) {
// //       throw new Error("Web Audio API is not supported in this browser.");
// //     }

// //     const ctx = new AudioContextClass();

// //     audioContextRef.current = ctx;

// //     await ctx.resume();

// //     const source = ctx.createMediaStreamSource(stream);

// //     sourceNodeRef.current = source;

// //     // --------------------------------------------------------
// //     // Phase-1 implementation.
// //     //
// //     // ScriptProcessor allows us to keep this implementation
// //     // entirely inside App.jsx.
// //     //
// //     // Later production version:
// //     // move this to AudioWorklet.
// //     // --------------------------------------------------------

// //     const processor = ctx.createScriptProcessor(4096, 1, 1);

// //     processorNodeRef.current = processor;

// //     // Mute processor output.
// //     // We only need incoming microphone samples.

// //     const mute = ctx.createGain();

// //     mute.gain.value = 0;

// //     muteGainRef.current = mute;

// //     source.connect(processor);

// //     processor.connect(mute);

// //     mute.connect(ctx.destination);

// //     processor.onaudioprocess = (event) => {
// //       // ----------------------------------------------------
// //       // Don't listen while:
// //       //
// //       // backend is processing
// //       // OR
// //       // assistant is speaking
// //       //
// //       // This prevents assistant TTS from feeding into STT.
// //       // ----------------------------------------------------

// //       if (
// //         !sessionActiveRef.current ||
// //         processingRef.current ||
// //         assistantSpeakingRef.current
// //       ) {
// //         return;
// //       }

// //       const input = event.inputBuffer.getChannelData(0);

// //       const frame = new Float32Array(input);

// //       const sampleRate = event.inputBuffer.sampleRate || ctx.sampleRate;

// //       const frameMs = (frame.length / sampleRate) * 1000;

// //       // ----------------------------------------------------
// //       // Calculate RMS microphone energy
// //       // ----------------------------------------------------

// //       let sumSquares = 0;

// //       for (let i = 0; i < frame.length; i++) {
// //         sumSquares += frame[i] * frame[i];
// //       }

// //       const rms = Math.sqrt(sumSquares / frame.length);

// //       const now = performance.now();

// //       // ====================================================
// //       // No speech yet
// //       // ====================================================

// //       if (!speechActiveRef.current) {
// //         pushPreRoll(frame, sampleRate);

// //         if (rms >= VAD_START_THRESHOLD) {
// //           speechActiveRef.current = true;

// //           speechStartedAtRef.current = now;

// //           speechEnergyMsRef.current = frameMs;

// //           silenceStartRef.current = null;

// //           // Include pre-roll to avoid clipping
// //           // beginning of sentence.

// //           utterancePcmRef.current = [...preRollRef.current];

// //           preRollRef.current = [];

// //           preRollSamplesRef.current = 0;

// //           setVoiceState("hearing");
// //         }

// //         return;
// //       }

// //       // ====================================================
// //       // Speech active
// //       // ====================================================

// //       utterancePcmRef.current.push(frame);

// //       if (rms >= VAD_END_THRESHOLD) {
// //         speechEnergyMsRef.current += frameMs;

// //         silenceStartRef.current = null;
// //       } else if (silenceStartRef.current == null) {
// //         silenceStartRef.current = now;
// //       }

// //       const utteranceMs = now - (speechStartedAtRef.current || now);

// //       const silentForMs =
// //         silenceStartRef.current == null ? 0 : now - silenceStartRef.current;

// //       // ====================================================
// //       // AUTO SUBMIT
// //       // ====================================================

// //       if (
// //         silentForMs >= VAD_SILENCE_MS ||
// //         utteranceMs >= VAD_MAX_UTTERANCE_MS
// //       ) {
// //         sendCurrentUtterance(sampleRate);
// //       }
// //     };
// //   }

// //   // ==========================================================
// //   // Start whole voice conversation
// //   // ==========================================================

// //   async function startVoiceSession() {
// //     if (sessionActiveRef.current) {
// //       return;
// //     }

// //     setError("");

// //     setCorrection("");

// //     setLiveTranscript("");

// //     setVoiceState("connecting");

// //     try {
// //       const stream = await navigator.mediaDevices.getUserMedia({
// //         audio: {
// //           echoCancellation: true,

// //           noiseSuppression: true,

// //           autoGainControl: true,

// //           channelCount: 1,
// //         },
// //       });

// //       micStreamRef.current = stream;

// //       sessionActiveRef.current = true;

// //       setSessionActive(true);

// //       // Connect persistent socket once.

// //       await connectVoiceSocket();

// //       // Keep mic alive and run VAD.

// //       await setupVoiceCapture(stream);

// //       resetVadBuffers();

// //       setVoiceState("listening");
// //     } catch (e) {
// //       setError(`Could not start voice session: ${e}`);

// //       cleanupVoiceResources(true);

// //       setSessionActive(false);

// //       setVoiceState("idle");

// //       setBusy(false);
// //     }
// //   }

// //   // ==========================================================
// //   // User manually ends whole conversation
// //   // ==========================================================

// //   function endVoiceSession() {
// //     cleanupVoiceResources(true);

// //     setSessionActive(false);

// //     setVoiceState("idle");

// //     setBusy(false);

// //     setLiveTranscript("");
// //   }

// //   // ==========================================================
// //   // Normal text query remains unchanged
// //   // ==========================================================

// //   async function submitText() {
// //     const query = textQuery.trim();

// //     if (!query) {
// //       return;
// //     }

// //     setBusy(true);

// //     setError("");

// //     setResult(null);

// //     setCorrection("");

// //     try {
// //       const r = await fetch(
// //         `${API}/api/v1/query`,

// //         {
// //           method: "POST",

// //           headers: {
// //             "Content-Type": "application/json",
// //           },

// //           body: JSON.stringify({
// //             query,

// //             language,

// //             top_k: 5,
// //           }),
// //         },
// //       );

// //       if (!r.ok) {
// //         throw new Error(await r.text());
// //       }

// //       const data = await r.json();

// //       setResult({
// //         ...data,

// //         transcript: query,
// //       });
// //     } catch (e) {
// //       setError(String(e));
// //     } finally {
// //       setBusy(false);
// //     }
// //   }

// //   // ==========================================================
// //   // Replay TTS manually
// //   // ==========================================================

// //   function speakAgain() {
// //     if (!result?.answer) {
// //       return;
// //     }

// //     // Gate microphone during replay.

// //     processingRef.current = true;

// //     assistantSpeakingRef.current = true;

// //     setVoiceState("speaking");

// //     playAssistantResult(result);
// //   }

// //   // ==========================================================
// //   // Feedback
// //   // ==========================================================

// //   async function feedback(rating) {
// //     if (!result?.request_id) {
// //       return;
// //     }

// //     const r = await fetch(
// //       `${API}/api/v1/feedback`,

// //       {
// //         method: "POST",

// //         headers: {
// //           "Content-Type": "application/json",
// //         },

// //         body: JSON.stringify({
// //           request_id: result.request_id,

// //           language,

// //           rating,

// //           query: result.transcript || textQuery,

// //           model_answer: result.answer,

// //           corrected_answer: correction.trim() || null,

// //           sources: result.sources || [],
// //         }),
// //       },
// //     );

// //     if (!r.ok) {
// //       setError(`Feedback failed: ${await r.text()}`);
// //     }
// //   }

// //   // ==========================================================
// //   // UI
// //   // ==========================================================

// //   return (
// //     <main className="page">
// //       <section className="card">
// //         <div className="eyebrow">VOICE + RAG</div>

// //         <h1>English + Hindi document assistant</h1>

// //         <p className="sub">
// //           Ask by voice or text. The app retrieves your English/Hindi documents,
// //           generates a grounded answer, and can speak the answer back.
// //         </p>

// //         {/* LANGUAGE */}

// //         <div className="language-row">
// //           <label htmlFor="language">Answer / speech language</label>

// //           <select
// //             id="language"
// //             value={language}
// //             onChange={(e) => setLanguage(e.target.value)}
// //           >
// //             <option value="en">English</option>

// //             <option value="hi">Hindi / हिन्दी</option>
// //           </select>
// //         </div>

// //         {/* ====================================================
// //             CONTINUOUS VOICE MODE
// //            ==================================================== */}

// //         <div className="mode-block">
// //           <div className="label">Voice conversation</div>

// //           <div className="controls">
// //             {!sessionActive ? (
// //               <button className="primary" onClick={startVoiceSession}>
// //                 <Mic size={18} />
// //                 Start conversation
// //               </button>
// //             ) : (
// //               <button className="danger" onClick={endVoiceSession}>
// //                 <Square size={18} />
// //                 End conversation
// //               </button>
// //             )}
// //           </div>

// //           <div className="ready">{voiceStateText(voiceState)}</div>

// //           {sessionActive && (
// //             <div className="ready">
// //               Speak normally. A pause of about {VAD_SILENCE_MS / 1000}s
// //               automatically sends the current question. The microphone resumes
// //               after the assistant finishes speaking.
// //             </div>
// //           )}

// //           {liveTranscript && sessionActive && (
// //             <div className="transcript">Heard: {liveTranscript}</div>
// //           )}
// //         </div>

// //         {/* ====================================================
// //             TEXT QUERY
// //            ==================================================== */}

// //         <div className="mode-block">
// //           <div className="label">Text query</div>

// //           <div className="text-query-row">
// //             <input
// //               value={textQuery}
// //               onChange={(e) => setTextQuery(e.target.value)}
// //               onKeyDown={(e) => {
// //                 if (e.key === "Enter") {
// //                   submitText();
// //                 }
// //               }}
// //               placeholder={
// //                 language === "hi" ? "अपना सवाल लिखें…" : "Type your question…"
// //               }
// //             />

// //             <button onClick={submitText} disabled={!textQuery.trim() || busy}>
// //               <MessageSquareText size={18} />
// //               Ask text
// //             </button>
// //           </div>
// //         </div>

// //         {/* ERROR */}

// //         {error && <div className="error">{error}</div>}

// //         {/* RESULT */}

// //         {result?.answer && (
// //           <div className="result">
// //             <div className="label">Question / transcript</div>

// //             <div className="transcript">{result.transcript}</div>

// //             <div className="label row">
// //               <span>Grounded answer</span>

// //               <button className="icon" onClick={speakAgain}>
// //                 <Volume2 size={18} />
// //               </button>
// //             </div>

// //             <div className="answer">{result.answer}</div>

// //             {/* Feedback */}

// //             <div className="feedback-block">
// //               <label>Optional human correction for future training</label>

// //               <textarea
// //                 value={correction}
// //                 onChange={(e) => setCorrection(e.target.value)}
// //                 placeholder={
// //                   "If the answer is wrong, type the correct answer before pressing thumbs down."
// //                 }
// //               />

// //               <div className="feedback">
// //                 <span>Was this grounded and useful?</span>

// //                 <button onClick={() => feedback(1)}>
// //                   <ThumbsUp size={16} />
// //                 </button>

// //                 <button onClick={() => feedback(-1)}>
// //                   <ThumbsDown size={16} />
// //                 </button>
// //               </div>
// //             </div>

// //             {/* Debug sources */}

// //             <details>
// //               <summary>Sources & latency</summary>

// //               <pre>
// //                 {JSON.stringify(
// //                   {
// //                     sources: result.sources,

// //                     timings_ms: result.timings_ms,
// //                   },
// //                   null,
// //                   2,
// //                 )}
// //               </pre>
// //             </details>
// //           </div>
// //         )}
// //       </section>
// //     </main>
// //   );
// // }

// import React, { useRef, useState } from "react";
// import {
//   Mic,
//   Square,
//   Send,
//   ThumbsUp,
//   ThumbsDown,
//   Volume2,
//   MessageSquareText,
// } from "lucide-react";

// const API = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

// const BROWSER_LANG = {
//   en: "en-IN",
//   hi: "hi-IN",
// };

// function pickMimeType() {
//   const options = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];
//   return options.find((t) => window.MediaRecorder?.isTypeSupported?.(t)) || "";
// }

// function base64ToBlobUrl(b64, mime = "audio/mpeg") {
//   const binary = atob(b64);
//   const bytes = new Uint8Array(binary.length);
//   for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
//   return URL.createObjectURL(new Blob([bytes], { type: mime }));
// }

// function browserSpeak(text, language) {
//   if (!("speechSynthesis" in window) || !text) return;
//   window.speechSynthesis.cancel();
//   const utter = new SpeechSynthesisUtterance(text);
//   utter.lang = BROWSER_LANG[language] || "en-IN";
//   window.speechSynthesis.speak(utter);
// }

// export default function App() {
//   const [recording, setRecording] = useState(false);
//   const [blob, setBlob] = useState(null);
//   const [busy, setBusy] = useState(false);
//   const [result, setResult] = useState(null);
//   const [error, setError] = useState("");
//   const [language, setLanguage] = useState("en");
//   const [textQuery, setTextQuery] = useState("");
//   const [correction, setCorrection] = useState("");
//   const mediaRecorder = useRef(null);
//   const chunks = useRef([]);

//   async function startRecording() {
//     setError("");
//     setResult(null);
//     setBlob(null);
//     setCorrection("");
//     try {
//       const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
//       const mimeType = pickMimeType();
//       const rec = mimeType
//         ? new MediaRecorder(stream, { mimeType })
//         : new MediaRecorder(stream);
//       chunks.current = [];
//       rec.ondataavailable = (e) => {
//         if (e.data.size) chunks.current.push(e.data);
//       };
//       rec.onstop = () => {
//         const out = new Blob(chunks.current, {
//           type: rec.mimeType || "audio/webm",
//         });
//         setBlob(out);
//         stream.getTracks().forEach((t) => t.stop());
//       };
//       mediaRecorder.current = rec;
//       rec.start();
//       setRecording(true);
//     } catch (e) {
//       setError(`Microphone access failed: ${e}`);
//     }
//   }

//   function stopRecording() {
//     mediaRecorder.current?.stop();
//     setRecording(false);
//   }

//   async function submitVoice() {
//     if (!blob) return;
//     setBusy(true);
//     setError("");
//     try {
//       const ext = blob.type.includes("mp4") ? "mp4" : "webm";
//       const form = new FormData();
//       form.append(
//         "audio",
//         new File([blob], `speech.${ext}`, { type: blob.type }),
//       );
//       form.append("language", language);
//       form.append("top_k", "5");
//       const r = await fetch(`${API}/api/v1/voice-query`, {
//         method: "POST",
//         body: form,
//       });
//       if (!r.ok) throw new Error(await r.text());
//       const data = await r.json();
//       setResult(data);
//       if (data.audio_base64) {
//         const url = base64ToBlobUrl(data.audio_base64, data.audio_mime);
//         new Audio(url).play();
//       } else if (data.browser_tts_fallback) {
//         browserSpeak(data.answer, language);
//       }
//     } catch (e) {
//       setError(String(e));
//     } finally {
//       setBusy(false);
//     }
//   }

//   async function submitText() {
//     const query = textQuery.trim();
//     if (!query) return;
//     setBusy(true);
//     setError("");
//     setResult(null);
//     setCorrection("");
//     try {
//       const r = await fetch(`${API}/api/v1/query`, {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ query, language, top_k: 5 }),
//       });
//       if (!r.ok) throw new Error(await r.text());
//       const data = await r.json();
//       setResult({ ...data, transcript: query });
//     } catch (e) {
//       setError(String(e));
//     } finally {
//       setBusy(false);
//     }
//   }

//   function speakAgain() {
//     if (!result) return;
//     if (result.audio_base64) {
//       new Audio(base64ToBlobUrl(result.audio_base64, result.audio_mime)).play();
//     } else {
//       browserSpeak(result.answer, language);
//     }
//   }

//   async function feedback(rating) {
//     if (!result) return;
//     const r = await fetch(`${API}/api/v1/feedback`, {
//       method: "POST",
//       headers: { "Content-Type": "application/json" },
//       body: JSON.stringify({
//         request_id: result.request_id,
//         language,
//         rating,
//         query: result.transcript || textQuery,
//         model_answer: result.answer,
//         corrected_answer: correction.trim() || null,
//         sources: result.sources,
//       }),
//     });
//     if (!r.ok) setError(`Feedback failed: ${await r.text()}`);
//   }

//   return (
//     <main className="page">
//       <section className="card">
//         <div className="eyebrow">VOICE + RAG</div>
//         <h1>English + Hindi document assistant</h1>
//         <p className="sub">
//           Ask by voice or text. The app retrieves your English/Hindi documents,
//           generates a grounded answer, and can speak the answer back.
//         </p>

//         <div className="language-row">
//           <label htmlFor="language">Answer / speech language</label>
//           <select
//             id="language"
//             value={language}
//             onChange={(e) => setLanguage(e.target.value)}
//           >
//             <option value="en">English</option>
//             <option value="hi">Hindi / हिन्दी</option>
//           </select>
//         </div>

//         <div className="mode-block">
//           <div className="label">Voice query</div>
//           <div className="controls">
//             {!recording ? (
//               <button className="primary" onClick={startRecording}>
//                 <Mic size={18} /> Record
//               </button>
//             ) : (
//               <button className="danger" onClick={stopRecording}>
//                 <Square size={18} /> Stop
//               </button>
//             )}
//             <button onClick={submitVoice} disabled={!blob || busy}>
//               <Send size={18} />
//               {busy ? "Working…" : "Ask with voice"}
//             </button>
//           </div>
//           {blob && (
//             <div className="ready">
//               Audio captured: {(blob.size / 1024).toFixed(1)} KB
//             </div>
//           )}
//         </div>

//         <div className="mode-block">
//           <div className="label">Text query</div>
//           <div className="text-query-row">
//             <input
//               value={textQuery}
//               onChange={(e) => setTextQuery(e.target.value)}
//               onKeyDown={(e) => {
//                 if (e.key === "Enter") submitText();
//               }}
//               placeholder={
//                 language === "hi" ? "अपना सवाल लिखें…" : "Type your question…"
//               }
//             />
//             <button onClick={submitText} disabled={!textQuery.trim() || busy}>
//               <MessageSquareText size={18} />
//               Ask text
//             </button>
//           </div>
//         </div>

//         {error && <div className="error">{error}</div>}

//         {result && (
//           <div className="result">
//             <div className="label">Question / transcript</div>
//             <div className="transcript">{result.transcript}</div>
//             <div className="label row">
//               <span>Grounded answer</span>
//               <button className="icon" onClick={speakAgain}>
//                 <Volume2 size={18} />
//               </button>
//             </div>
//             <div className="answer">{result.answer}</div>
//             <div className="feedback-block">
//               <label>Optional human correction for future training</label>
//               <textarea
//                 value={correction}
//                 onChange={(e) => setCorrection(e.target.value)}
//                 placeholder="If the answer is wrong, type the correct answer before pressing thumbs down."
//               />
//               <div className="feedback">
//                 <span>Was this grounded and useful?</span>
//                 <button onClick={() => feedback(1)}>
//                   <ThumbsUp size={16} />
//                 </button>
//                 <button onClick={() => feedback(-1)}>
//                   <ThumbsDown size={16} />
//                 </button>
//               </div>
//             </div>
//             <details>
//               <summary>Sources & latency</summary>
//               <pre>
//                 {JSON.stringify(
//                   { sources: result.sources, timings_ms: result.timings_ms },
//                   null,
//                   2,
//                 )}
//               </pre>
//             </details>
//           </div>
//         )}
//       </section>
//     </main>
//   );
// }

import React, { useRef, useState } from "react";

import {
  Mic,
  Square,
  Send,
  ThumbsUp,
  ThumbsDown,
  Volume2,
  MessageSquareText,
} from "lucide-react";

const API = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const BROWSER_LANG = {
  en: "en-IN",
  hi: "hi-IN",
};

function pickMimeType() {
  const options = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];

  return (
    options.find((type) => window.MediaRecorder?.isTypeSupported?.(type)) || ""
  );
}

function base64ToBlobUrl(b64, mime = "audio/mpeg") {
  const binary = atob(b64);

  const bytes = new Uint8Array(binary.length);

  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }

  return URL.createObjectURL(
    new Blob([bytes], {
      type: mime,
    }),
  );
}

function browserSpeak(text, language) {
  if (!("speechSynthesis" in window) || !text) {
    return;
  }

  window.speechSynthesis.cancel();

  const utter = new SpeechSynthesisUtterance(text);

  utter.lang = BROWSER_LANG[language] || "en-IN";

  window.speechSynthesis.speak(utter);
}

export default function App() {
  const [recording, setRecording] = useState(false);

  const [blob, setBlob] = useState(null);

  const [busy, setBusy] = useState(false);

  const [result, setResult] = useState(null);

  const [error, setError] = useState("");

  const [language, setLanguage] = useState("en");

  // =========================================================
  // NEW:
  //
  // true  -> Document RAG
  // false -> Direct LLM
  //
  // Default true preserves original behavior.
  // =========================================================

  const [ragBased, setRagBased] = useState(true);

  const [textQuery, setTextQuery] = useState("");

  const [correction, setCorrection] = useState("");

  const mediaRecorder = useRef(null);

  const chunks = useRef([]);

  // =========================================================
  // RECORD AUDIO
  // =========================================================

  async function startRecording() {
    setError("");

    setResult(null);

    setBlob(null);

    setCorrection("");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
      });

      const mimeType = pickMimeType();

      const rec = mimeType
        ? new MediaRecorder(stream, {
            mimeType,
          })
        : new MediaRecorder(stream);

      chunks.current = [];

      rec.ondataavailable = (event) => {
        if (event.data.size) {
          chunks.current.push(event.data);
        }
      };

      rec.onstop = () => {
        const out = new Blob(chunks.current, {
          type: rec.mimeType || "audio/webm",
        });

        setBlob(out);

        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.current = rec;

      rec.start();

      setRecording(true);
    } catch (e) {
      setError(`Microphone access failed: ${e}`);
    }
  }

  // =========================================================
  // STOP AUDIO
  // =========================================================

  function stopRecording() {
    mediaRecorder.current?.stop();

    setRecording(false);
  }

  // =========================================================
  // VOICE QUERY
  // =========================================================

  async function submitVoice() {
    if (!blob) {
      return;
    }

    setBusy(true);

    setError("");

    try {
      const ext = blob.type.includes("mp4") ? "mp4" : "webm";

      const form = new FormData();

      form.append(
        "audio",

        new File(
          [blob],

          `speech.${ext}`,

          {
            type: blob.type,
          },
        ),
      );

      form.append("language", language);

      form.append("top_k", "5");

      // =====================================================
      // NEW
      // =====================================================

      form.append("rag_based", String(ragBased));

      const response = await fetch(
        `${API}/api/v1/voice-query`,

        {
          method: "POST",

          body: form,
        },
      );

      if (!response.ok) {
        throw new Error(await response.text());
      }

      const data = await response.json();

      setResult(data);

      // =====================================================
      // Voice response
      // =====================================================

      if (data.audio_base64) {
        const url = base64ToBlobUrl(
          data.audio_base64,

          data.audio_mime,
        );

        new Audio(url).play();
      } else if (data.browser_tts_fallback) {
        browserSpeak(data.answer, language);
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  // =========================================================
  // TEXT QUERY
  // =========================================================

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
      const response = await fetch(
        `${API}/api/v1/query`,

        {
          method: "POST",

          headers: {
            "Content-Type": "application/json",
          },

          body: JSON.stringify({
            query,

            language,

            top_k: 5,

            // ============================================
            // NEW
            // ============================================

            rag_based: ragBased,
          }),
        },
      );

      if (!response.ok) {
        throw new Error(await response.text());
      }

      const data = await response.json();

      setResult({
        ...data,

        transcript: query,
      });
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  // =========================================================
  // REPLAY SPEECH
  // =========================================================

  function speakAgain() {
    if (!result) {
      return;
    }

    if (result.audio_base64) {
      new Audio(base64ToBlobUrl(result.audio_base64, result.audio_mime)).play();
    } else {
      browserSpeak(result.answer, language);
    }
  }

  // =========================================================
  // HUMAN FEEDBACK
  // =========================================================

  async function feedback(rating) {
    if (!result) {
      return;
    }

    const response = await fetch(
      `${API}/api/v1/feedback`,

      {
        method: "POST",

        headers: {
          "Content-Type": "application/json",
        },

        body: JSON.stringify({
          request_id: result.request_id,

          language,

          rating,

          query: result.transcript || textQuery,

          model_answer: result.answer,

          corrected_answer: correction.trim() || null,

          sources: result.sources || [],

          // ==============================================
          // NEW:
          // Preserve which pipeline user evaluated.
          // ==============================================

          rag_based: result.rag_based ?? ragBased,
        }),
      },
    );

    if (!response.ok) {
      setError(`Feedback failed: ${await response.text()}`);
    }
  }

  // =========================================================
  // UI
  // =========================================================

  return (
    <main className="page">
      <section className="card">
        <div className="eyebrow">VOICE + RAG</div>

        <h1>English + Hindi document assistant</h1>

        <p className="sub">
          Ask by voice or text. Use Document RAG mode for answers grounded in
          your uploaded documents, or Direct LLM mode for general model
          knowledge.
        </p>

        {/* ================================================
            LANGUAGE
           ================================================ */}

        <div className="language-row">
          <label htmlFor="language">Answer / speech language</label>

          <select
            id="language"
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
          >
            <option value="en">English</option>

            <option value="hi">Hindi / हिन्दी</option>
          </select>
        </div>

        {/* ================================================
            ANSWER MODE
           ================================================ */}

        <div className="language-row">
          <label htmlFor="answer-mode">Answer mode</label>

          <select
            id="answer-mode"
            value={ragBased ? "rag" : "direct"}
            onChange={(e) => {
              setRagBased(e.target.value === "rag");
            }}
          >
            <option value="rag">Document RAG</option>

            <option value="direct">Direct LLM</option>
          </select>
        </div>

        <div className="ready">
          {ragBased
            ? "Document RAG: retrieve relevant chunks " +
              "from your uploaded knowledge base before answering."
            : "Direct LLM: skip document retrieval and " +
              "send the question directly to the LLM."}
        </div>

        {/* ================================================
            VOICE QUERY
           ================================================ */}

        <div className="mode-block">
          <div className="label">Voice query</div>

          <div className="controls">
            {!recording ? (
              <button className="primary" onClick={startRecording}>
                <Mic size={18} />
                Record
              </button>
            ) : (
              <button className="danger" onClick={stopRecording}>
                <Square size={18} />
                Stop
              </button>
            )}

            <button onClick={submitVoice} disabled={!blob || busy}>
              <Send size={18} />

              {busy ? "Working…" : "Ask with voice"}
            </button>
          </div>

          {blob && (
            <div className="ready">
              Audio captured: {(blob.size / 1024).toFixed(1)} KB
            </div>
          )}
        </div>

        {/* ================================================
            TEXT QUERY
           ================================================ */}

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
              placeholder={
                language === "hi" ? "अपना सवाल लिखें…" : "Type your question…"
              }
            />

            <button onClick={submitText} disabled={!textQuery.trim() || busy}>
              <MessageSquareText size={18} />
              Ask text
            </button>
          </div>
        </div>

        {/* ================================================
            ERROR
           ================================================ */}

        {error && <div className="error">{error}</div>}

        {/* ================================================
            RESULT
           ================================================ */}

        {result && (
          <div className="result">
            <div className="ready">
              Mode: {result.rag_based ? "Document RAG" : "Direct LLM"}
            </div>

            <div className="label">Question / transcript</div>

            <div className="transcript">{result.transcript}</div>

            <div className="label row">
              <span>
                {result.rag_based ? "Grounded answer" : "Direct LLM answer"}
              </span>

              <button className="icon" onClick={speakAgain}>
                <Volume2 size={18} />
              </button>
            </div>

            <div className="answer">{result.answer}</div>

            {/* ============================================
                HUMAN FEEDBACK
               ============================================ */}

            <div className="feedback-block">
              <label>Optional human correction for future training</label>

              <textarea
                value={correction}
                onChange={(e) => setCorrection(e.target.value)}
                placeholder={
                  "If the answer is wrong, type the correct " +
                  "answer before pressing thumbs down."
                }
              />

              <div className="feedback">
                <span>
                  {result.rag_based
                    ? "Was this grounded and useful?"
                    : "Was this direct LLM answer useful?"}
                </span>

                <button onClick={() => feedback(1)}>
                  <ThumbsUp size={16} />
                </button>

                <button onClick={() => feedback(-1)}>
                  <ThumbsDown size={16} />
                </button>
              </div>
            </div>

            {/* ============================================
                DEBUG / EVALUATION
               ============================================ */}

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
