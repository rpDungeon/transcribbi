// AudioWorklet processor: resample to 16kHz and convert to s16le
class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const samples = input[0]; // Float32Array, mono channel
    const inputRate = sampleRate; // Global in AudioWorklet scope
    const outputRate = 16000;
    const ratio = outputRate / inputRate;

    // Resample using linear interpolation
    const outputLen = Math.floor(samples.length * ratio);
    const s16 = new Int16Array(outputLen);

    for (let i = 0; i < outputLen; i++) {
      const srcPos = i / ratio;
      const idx = Math.floor(srcPos);
      const frac = srcPos - idx;

      let val;
      if (idx + 1 < samples.length) {
        val = samples[idx] * (1 - frac) + samples[idx + 1] * frac;
      } else {
        val = idx < samples.length ? samples[idx] : 0;
      }

      // Clamp and convert to Int16
      val = Math.max(-1, Math.min(1, val));
      s16[i] = val < 0 ? val * 0x8000 : val * 0x7FFF;
    }

    // Post the s16le buffer to main thread
    this.port.postMessage(s16.buffer, [s16.buffer]);
    return true;
  }
}

registerProcessor("pcm-processor", PCMProcessor);
