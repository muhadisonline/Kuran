import test from 'node:test';
import assert from 'node:assert/strict';
import {
  computeAdaptiveGain,
  createInitialDspState,
  processVoiceSample,
  normalizeArabicText,
  arabicWordSimilarity,
  matchSpokenArabicToAyahWords,
} from './voiceDsp';

test('computeAdaptiveGain gates out ambient room silence and does not trigger speech activity', () => {
  // In silence / ambient hiss (e.g. 0.00015 RMS), speech must NOT be active and targetGain stays at 1.5
  const silence = computeAdaptiveGain(0.00015, 4.0, 'çox_güclü');
  assert.equal(silence.isSpeechActive, false);
  assert.equal(silence.targetGain, 1.5);
  assert.ok(silence.nextGain < 4.0, 'Gain should relax downward during silence');
});

test('computeAdaptiveGain strongly amplifies genuine quiet voice above noise gate', () => {
  // At 0.0012 RMS (quiet human recitation), DSP actively boosts
  const res = computeAdaptiveGain(0.0012, 4.0, 'çox_güclü');
  assert.equal(res.isSpeechActive, true);
  assert.ok(res.targetGain > 20, `Expected targetGain > 20 for quiet voice, got ${res.targetGain}`);
  assert.ok(res.nextGain > 4.0, `Expected nextGain to increase from 4.0, got ${res.nextGain}`);
});

test('computeAdaptiveGain ultra mode provides higher boost than strong mode', () => {
  const strong = computeAdaptiveGain(0.002, 4.0, 'güclü');
  const ultra = computeAdaptiveGain(0.002, 4.0, 'ultra');
  assert.ok(ultra.targetGain > strong.targetGain);
  assert.ok(ultra.dBBoost > strong.dBBoost);
});

test('processVoiceSample amplifies voice while clamping loud peaks within [-1, 1]', () => {
  const state = createInitialDspState();
  let quietOut = 0;
  for (let i = 0; i < 300; i++) {
    const raw = Math.sin(i * 0.4) * 0.004;
    quietOut = processVoiceSample(raw, state, 'ultra');
  }
  assert.ok(Math.abs(quietOut) > 0.004, 'Quiet sinusoidal voice signal should be amplified');

  // Loud signal should never clip outside [-1, 1]
  const loudOut = processVoiceSample(0.95, state, 'ultra');
  assert.ok(loudOut <= 1.0 && loudOut >= -1.0);
});

test('normalizeArabicText strips tashkeel and normalizes Alef, Ya, and Ta Marbuta', () => {
  const fatiha1 = 'بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ';
  const norm = normalizeArabicText(fatiha1);
  assert.equal(norm, 'بسم الله الرحمن الرحيم');
});

test('arabicWordSimilarity handles Uthmani vs standard Arabic spelling variations', () => {
  assert.ok(arabicWordSimilarity('ٱلْعَـٰلَمِينَ', 'العالمين') >= 0.8);
  assert.ok(arabicWordSimilarity('ٱلصِّرَٰطَ', 'الصراط') >= 0.8);
  assert.ok(arabicWordSimilarity('مَـٰلِكِ', 'مالك') >= 0.75);
});

test('matchSpokenArabicToAyahWords matches sequential words and rejects non-matching words', () => {
  const ayahWords = ['بِسْمِ', 'ٱللَّهِ', 'ٱلرَّحْمَـٰنِ', 'ٱلرَّحِيمِ'];
  const matched = matchSpokenArabicToAyahWords('بسم الله الرحمن', ayahWords, 0);
  assert.deepEqual(matched, [0, 1, 2]);

  const wrong = matchSpokenArabicToAyahWords('كتاب كبير', ayahWords, 0);
  assert.deepEqual(wrong, []);
});
