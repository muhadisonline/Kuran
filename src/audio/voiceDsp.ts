/**
 * Pure DSP & Arabic Recitation Matching Utilities
 * Decoupled for deterministic unit testing and runtime audio enhancement.
 *
 * Key invariant:
 * - Ambient silence / room noise below `noiseGateRms` is NEVER amplified into false speech
 *   (`isSpeechActive = false`, gain relaxes to a low resting level <= 2.5x).
 * - Actual human voice above `noiseGateRms` is cleanly amplified up to `maxAdaptiveGain`
 *   to reach `targetRms` without distorting log-mel spectrogram features.
 */

export type VoiceBoostMode = 'güclü' | 'çox_güclü' | 'ultra';

export interface VoiceBoostConfig {
  mode: VoiceBoostMode;
  labelAz: string;
  preAmpGain: number;
  workletBoost: number;
  maxAdaptiveGain: number;
  noiseGateRms: number;
  targetRms: number;
}

export const VOICE_BOOST_PRESETS: Record<VoiceBoostMode, VoiceBoostConfig> = {
  güclü: {
    mode: 'güclü',
    labelAz: 'Güclü (2x)',
    preAmpGain: 1.5,
    workletBoost: 1.4,
    maxAdaptiveGain: 24,
    noiseGateRms: 0.00075,
    targetRms: 0.13,
  },
  çox_güclü: {
    mode: 'çox_güclü',
    labelAz: 'Çox güclü (4x)',
    preAmpGain: 2.2,
    workletBoost: 2.0,
    maxAdaptiveGain: 36,
    noiseGateRms: 0.00050,
    targetRms: 0.15,
  },
  ultra: {
    mode: 'ultra',
    labelAz: 'Ultra (8x)',
    preAmpGain: 3.0,
    workletBoost: 2.8,
    maxAdaptiveGain: 48,
    noiseGateRms: 0.00035,
    targetRms: 0.17,
  },
};

export interface AdaptiveGainResult {
  nextGain: number;
  targetGain: number;
  dBBoost: number;
  isSpeechActive: boolean;
}

/**
 * Computes next adaptive gain step for real-time Quranic voice capture.
 * - When `rms < noiseGateRms` (silence or quiet room background), `isSpeechActive` is false
 *   and `targetGain` drops to a low resting gain (1.5x) so background hiss is never amplified.
 * - When `rms >= noiseGateRms` (user is speaking), `targetGain` scales up to `maxAdaptiveGain`.
 */
export function computeAdaptiveGain(
  rms: number,
  currentGain: number,
  mode: VoiceBoostMode = 'çox_güclü'
): AdaptiveGainResult {
  const preset = VOICE_BOOST_PRESETS[mode] ?? VOICE_BOOST_PRESETS.çox_güclü;
  const safeRms = Math.max(0, Number.isFinite(rms) ? rms : 0);
  const isSpeechActive = safeRms >= preset.noiseGateRms;

  const rawTarget = isSpeechActive
    ? Math.min(preset.maxAdaptiveGain, (preset.targetRms / Math.max(0.0001, safeRms)) * (preset.workletBoost * 0.6))
    : 1.5;

  const targetGain = isSpeechActive
    ? Math.max(1.5, Math.min(preset.maxAdaptiveGain, rawTarget))
    : 1.5;

  const rate = targetGain > currentGain ? 0.035 : 0.004;
  const nextGain = currentGain + (targetGain - currentGain) * rate;
  const effectiveTotalGain = isSpeechActive ? nextGain * preset.preAmpGain : nextGain;
  const dBBoost = Math.round(20 * Math.log10(Math.max(1, effectiveTotalGain)) * 10) / 10;

  return {
    nextGain,
    targetGain,
    dBBoost,
    isSpeechActive,
  };
}

export interface DspFilterState {
  x1: number;
  y1: number;
  env: number;
  gain: number;
}

export function createInitialDspState(): DspFilterState {
  return {
    x1: 0,
    y1: 0,
    env: 0.0001,
    gain: 2.0,
  };
}

/**
 * Processes a single 16kHz PCM audio sample through 80Hz high-pass,
 * noise-gated adaptive envelope AGC, and linear clamp [-1, 1].
 */
export function processVoiceSample(
  sample: number,
  state: DspFilterState,
  mode: VoiceBoostMode = 'çox_güclü'
): number {
  const hpCoeff = Math.exp((-2 * Math.PI * 80) / 16000);
  const hpOut = hpCoeff * (state.y1 + sample - state.x1);
  state.x1 = sample;
  state.y1 = hpOut;

  const instSq = hpOut * hpOut;
  const envAlpha = instSq > state.env ? 0.003 : 0.0007;
  state.env = state.env * (1 - envAlpha) + instSq * envAlpha;

  const rms = Math.sqrt(state.env);
  const { nextGain } = computeAdaptiveGain(rms, state.gain, mode);
  state.gain = nextGain;

  const absHp = Math.abs(hpOut);
  if (absHp * state.gain > 0.88 && absHp > 1e-5) {
    state.gain = Math.max(1.0, 0.88 / absHp);
  }

  const v = hpOut * state.gain;
  return v > 1 ? 1 : v < -1 ? -1 : v;
}

/**
 * Strips Arabic diacritics (tashkeel), Quranic stop marks, tatweel,
 * and normalizes alef/hamza/ya/ta-marbuta for resilient word matching.
 */
export function normalizeArabicText(input: string): string {
  if (!input) return '';
  return input
    // Remove Quranic annotations, harakat, diacritics, superscript alef
    .replace(/[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640\u08F0-\u08FF]/g, '')
    // Normalize Alef variants (أ إ آ ٱ) to bare Alef (ا)
    .replace(/[أإآٱ]/g, 'ا')
    // Normalize Alef Maqsura (ى) and Farsi Yeh (ی) to Yeh (ي)
    .replace(/[ىی]/g, 'ي')
    // Normalize Ta Marbuta (ة) to Ha (ه)
    .replace(/ة/g, 'ه')
    // Remove non-Arabic letters except spaces
    .replace(/[^\u0621-\u064A\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

/**
 * Computes normalized character similarity [0..1] between two normalized Arabic words.
 */
export function arabicWordSimilarity(a: string, b: string): number {
  const na = normalizeArabicText(a).replace(/\s+/g, '');
  const nb = normalizeArabicText(b).replace(/\s+/g, '');
  if (!na || !nb) return 0;
  if (na === nb) return 1;
  if (na.startsWith(nb) || nb.startsWith(na)) {
    const minLen = Math.min(na.length, nb.length);
    const maxLen = Math.max(na.length, nb.length);
    if (minLen >= 2) return Math.max(0.75, minLen / maxLen);
  }

  const m = na.length;
  const n = nb.length;
  const dp: number[] = Array.from({ length: n + 1 }, (_, j) => j);

  for (let i = 1; i <= m; i++) {
    let prev = dp[0];
    dp[0] = i;
    for (let j = 1; j <= n; j++) {
      const temp = dp[j];
      const cost = na[i - 1] === nb[j - 1] ? 0 : 1;
      dp[j] = Math.min(dp[j] + 1, dp[j - 1] + 1, prev + cost);
      prev = temp;
    }
  }

  const dist = dp[n];
  return Math.max(0, 1 - dist / Math.max(m, n));
}

/**
 * Matches spoken Arabic transcript words against expected ayah words starting at currentWordIndex,
 * returning all newly matched word indices (0-based).
 * Uses strict sequential matching from currentWordIndex (max lookahead 1 word) so unrelated
 * speech or noise never jumps ahead.
 */
export function matchSpokenArabicToAyahWords(
  spokenTranscript: string,
  expectedAyahWords: string[],
  currentWordIndex: number = 0,
  similarityThreshold: number = 0.78
): number[] {
  const spokenWords = normalizeArabicText(spokenTranscript)
    .split(' ')
    .filter(Boolean);
  if (spokenWords.length === 0 || expectedAyahWords.length === 0) {
    return [];
  }

  const matchedIndices: number[] = [];
  let searchStart = Math.max(0, currentWordIndex);

  for (const sw of spokenWords) {
    if (searchStart >= expectedAyahWords.length) break;
    // Only match the immediate expected word (or +1 if >= 3 chars and very high similarity)
    const simCurrent = arabicWordSimilarity(sw, expectedAyahWords[searchStart]);
    if (simCurrent >= similarityThreshold) {
      matchedIndices.push(searchStart);
      searchStart += 1;
      continue;
    }
    if (
      searchStart + 1 < expectedAyahWords.length &&
      sw.length >= 3 &&
      arabicWordSimilarity(sw, expectedAyahWords[searchStart + 1]) >= 0.85
    ) {
      matchedIndices.push(searchStart, searchStart + 1);
      searchStart += 2;
    }
  }

  return matchedIndices;
}
