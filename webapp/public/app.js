let audioCtx = null;
let workletNode = null;
let mediaStream = null;
let ws = null;
let recording = false;

const btn = document.getElementById("btn");
const output = document.getElementById("output");
const statusEl = document.getElementById("status");

function setStatus(msg) {
  statusEl.textContent = msg;
}

async function startRecording() {
  output.value = "";
  setStatus("Requesting microphone...");

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: 48000,
        echoCancellation: true,
        noiseSuppression: true,
      },
    });
  } catch (e) {
    setStatus("Microphone access denied: " + e.message);
    return;
  }

  // Connect WebSocket
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/ws`);
  ws.binaryType = "arraybuffer";

  ws.onopen = async () => {
    setStatus("Connected. Loading audio processor...");

    audioCtx = new AudioContext();
    await audioCtx.audioWorklet.addModule("processor.js");

    const source = audioCtx.createMediaStreamSource(mediaStream);
    workletNode = new AudioWorkletNode(audioCtx, "pcm-processor");

    // Worklet posts s16le buffers → send to server
    workletNode.port.onmessage = (e) => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(e.data);
      }
    };

    source.connect(workletNode);
    // Don't connect to destination (no playback)

    recording = true;
    btn.textContent = "Stop Recording";
    btn.classList.add("recording");
    setStatus("Recording... speak into your microphone");
  };

  ws.onmessage = (e) => {
    if (typeof e.data === "string") {
      if (e.data === "\n[END]") {
        setStatus("Transcription complete.");
        return;
      }
      output.value += e.data;
      output.scrollTop = output.scrollHeight;
    }
  };

  ws.onclose = () => {
    if (recording) {
      setStatus("Connection closed.");
      stopRecording(false);
    }
  };

  ws.onerror = (e) => {
    setStatus("WebSocket error. Is the server running?");
    console.error("WebSocket error:", e);
  };
}

function stopRecording(closeWs = true) {
  recording = false;
  btn.textContent = "Start Recording";
  btn.classList.remove("recording");

  if (workletNode) {
    workletNode.disconnect();
    workletNode = null;
  }
  if (audioCtx) {
    audioCtx.close();
    audioCtx = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
  if (closeWs && ws) {
    ws.close();
    ws = null;
  }
  if (!closeWs) {
    ws = null;
  }

  setStatus("Stopped. Waiting for remaining transcription...");
}

btn.addEventListener("click", () => {
  if (!recording) {
    startRecording();
  } else {
    stopRecording();
  }
});
