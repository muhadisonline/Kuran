/**
 * High-Sensitivity Quran Recitation AudioWorkletProcessor (mic-tap)
 *
 * Designed specifically for the Zipformer ONNX CTC Quran model:
 * - Preserves natural 16kHz spectral envelope (no nonlinear tanh distortion or artificial pre-emphasis)
 *   so 80-bin log-mel filterbanks match model training distribution.
 * - Noise gate keeps ambient room silence quiet (gain decays toward 1.0 during silence)
 *   so the model never hallucinates or advances on its own.
 * - Clean fast-attack AGC boosts genuine quiet/distant recitation up to maxGain (24x–48x).
 */
class MicTap extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.pos = 0;
    this.lp = 0;
    this.x1 = 0;
    this.y1 = 0;
    // 80Hz high-pass filter to remove DC offset and low-frequency room rumble
    this.hp = Math.exp((-2 * Math.PI * 80) / 16000);
    this.env = 0.0004;
    this.gain = 1.5;
    this.roomGain = 4.0;
    this.maxGain = 36.0;
    this.gate = 0.00055;
    this.targetRms = 0.14;
    this.energy = 0;
    this.buf = new Float32Array(4800); // 300ms frame at 16kHz
    this.n = 0;
    this.lpA = 1 - Math.exp((-2 * Math.PI * 7200) / sampleRate);

    this.port.onmessage = (ev) => {
      const data = ev.data;
      if (!data) return;
      if (typeof data.room === "number" && data.room >= 1 && data.room <= 48) {
        this.roomGain = data.room;
        this.gain = Math.min(this.maxGain, Math.max(1.5, data.room));
      }
      if (typeof data.maxGain === "number" && data.maxGain >= 10 && data.maxGain <= 64) {
        this.maxGain = data.maxGain;
      }
      if (typeof data.gate === "number" && data.gate >= 0.0001 && data.gate <= 0.01) {
        this.gate = data.gate;
      }
      if (typeof data.targetRms === "number" && data.targetRms >= 0.06 && data.targetRms <= 0.25) {
        this.targetRms = data.targetRms;
      }
    };
  }

  push(x) {
    // 80Hz high-pass filter
    const y = this.hp * (this.y1 + x - this.x1);
    this.x1 = x;
    this.y1 = y;

    // Accumulate true raw energy for accurate room calibration & level meter
    this.energy += y * y;
    const sq = y * y;
    const envAlpha = sq > this.env ? 0.0025 : 0.0005;
    this.env = this.env * (1 - envAlpha) + sq * envAlpha;

    const rms = Math.sqrt(this.env);
    // Only boost when signal is above the voice activity noise gate;
    // when silent, relax gain so room hiss is never amplified into false speech
    const isVoice = rms > this.gate;
    const desiredGain = isVoice
      ? Math.min(this.maxGain, Math.max(1.0, this.targetRms / rms))
      : Math.min(2.0, Math.max(1.0, this.roomGain * 0.25));
    const rate = isVoice
      ? desiredGain > this.gain
        ? 0.015
        : 0.0025
      : 0.002;
    this.gain += (desiredGain - this.gain) * rate;

    // Instant peak limiter prevents hard-clipping square-wave distortion on loud Qari audio
    const absY = Math.abs(y);
    if (absY * this.gain > 0.88 && absY > 1e-5) {
      this.gain = Math.max(1.0, 0.88 / absY);
    }

    const s = y * this.gain;
    this.buf[this.n] = s > 1 ? 1 : s < -1 ? -1 : s;
    this.n += 1;

    if (this.n === this.buf.length) {
      const out = this.buf;
      const rawRms = Math.sqrt(this.energy / this.buf.length);
      this.buf = new Float32Array(4800);
      this.n = 0;
      this.energy = 0;
      this.port.postMessage({ rms: rawRms, pcm: out }, [out.buffer]);
    }
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    if (this.ratio <= 1.02) {
      for (let i = 0; i < ch.length; i += 1) this.push(ch[i]);
      return true;
    }
    for (let i = 0; i < ch.length; i += 1) {
      this.lp += this.lpA * (ch[i] - this.lp);
      this.pos += 1;
      if (this.pos + 1e-4 < this.ratio) continue;
      this.pos -= this.ratio;
      this.push(this.lp);
    }
    return true;
  }
}

registerProcessor("mic-tap", MicTap);
