import test from 'node:test';
import assert from 'node:assert/strict';
import {
  detectHarakatMismatch,
  detectMakhrajConfusion,
  analyzeSpokenVsExpected,
  evaluateVerdictMistakes,
  alignWords,
  judgeAttemptTilawi,
  overlapMerge,
  getReciterAyahAudioUrl,
} from './mistakeDetector';

test('detectMakhrajConfusion catches classical Arabic sister-letter mistakes (Sad/Sin, Ha/He, Qaf/Kaf, Dad/Dal)', () => {
  const sadSin = detectMakhrajConfusion('ٱلصِّرَٰطَ', 'السراط');
  assert.ok(sadSin, 'Should detect Sad -> Sin confusion');
  assert.equal(sadSin?.expectedLetter, 'ص');
  assert.equal(sadSin?.heardLetter, 'س');

  const haHe = detectMakhrajConfusion('ٱلْحَمْدُ', 'الهمد');
  assert.ok(haHe, 'Should detect Ha -> He confusion');
  assert.equal(haHe?.expectedLetter, 'ح');
  assert.equal(haHe?.heardLetter, 'ه');

  const qafKaf = detectMakhrajConfusion('ٱلْمُسْتَقِيمَ', 'المستكيم');
  assert.ok(qafKaf, 'Should detect Qaf -> Kaf confusion');
  assert.equal(qafKaf?.expectedLetter, 'ق');
  assert.equal(qafKaf?.heardLetter, 'ك');

  const dadDal = detectMakhrajConfusion('ٱلضَّآلِّينَ', 'الدالين');
  assert.ok(dadDal, 'Should detect Dad -> Dal confusion');
  assert.equal(dadDal?.expectedLetter, 'ض');
  assert.equal(dadDal?.heardLetter, 'د');
});

test('detectHarakatMismatch catches fatha/damma/kasra vowel errors on identical consonant skeletons', () => {
  // an'amta (أَنْعَمْتَ) vs an'amtu (أَنْعَمْتُ)
  const res = detectHarakatMismatch('أَنْعَمْتَ', 'أَنْعَمْتُ');
  assert.equal(res.hasError, true);
  assert.ok(res.descriptionAz.includes('Fəthə'), 'Should mention expected Fatha');
  assert.ok(res.descriptionAz.includes('Zəmmə'), 'Should mention heard Damma');

  // Identical harakat should not flag error
  const clean = detectHarakatMismatch('أَنْعَمْتَ', 'أَنْعَمْتَ');
  assert.equal(clean.hasError, false);
});

test('analyzeSpokenVsExpected flags word omission when reciter skips a word', () => {
  const fatiha2 = ['ٱلْحَمْدُ', 'لِلَّهِ', 'رَبِّ', 'ٱلْعَـٰلَمِينَ'];
  // User is at word 0 (الحمد), skips it and says لله (word 1)
  const out = analyzeSpokenVsExpected({
    surah: 1,
    ayah: 2,
    expectedWords: fatiha2,
    currentWordIndex: 0,
    spokenTranscript: 'لله',
  });
  assert.equal(out.mistakes.length, 1);
  assert.equal(out.mistakes[0].kind, 'possible_omission');
  assert.equal(out.mistakes[0].word, 0);
  assert.deepEqual(out.matchedIndices, [1]);
});

test('analyzeSpokenVsExpected flags wrong word substitution and makhraj errors immediately', () => {
  const fatiha6 = ['ٱهْدِنَا', 'ٱلصِّرَٰطَ', 'ٱلْمُسْتَقِيمَ'];
  // User is at word 1 (الصراط) and says السراط (Sin instead of Sad)
  const makhrajOut = analyzeSpokenVsExpected({
    surah: 1,
    ayah: 6,
    expectedWords: fatiha6,
    currentWordIndex: 1,
    spokenTranscript: 'السراط',
  });
  assert.equal(makhrajOut.mistakes.length, 1);
  assert.equal(makhrajOut.mistakes[0].kind, 'possible_substitution');
  assert.equal(makhrajOut.mistakes[0].subKind, 'makhraj');
  assert.equal(makhrajOut.mistakes[0].word, 1);
  assert.deepEqual(makhrajOut.matchedIndices, []);

  // User says a completely wrong word
  const wrongOut = analyzeSpokenVsExpected({
    surah: 1,
    ayah: 6,
    expectedWords: fatiha6,
    currentWordIndex: 1,
    spokenTranscript: 'الكوثر',
  });
  assert.equal(wrongOut.mistakes.length, 1);
  assert.equal(wrongOut.mistakes[0].kind, 'possible_substitution');
  assert.equal(wrongOut.mistakes[0].word, 1);
});

test('evaluateVerdictMistakes catches first-word, last-word, vowel, and omission mistakes in Zipformer verdicts', () => {
  const words = ['بِسْمِ', 'ٱللَّهِ', 'ٱلرَّحْمَـٰنِ', 'ٱلرَّحِيمِ'];
  const verdicts = [
    {
      surah: 1,
      ayah: 1,
      word: 0,
      state: 'wrong' as const,
      distance: 0.42,
      heardRatio: 0.8,
      margin: 0.35,
    },
    {
      surah: 1,
      ayah: 1,
      word: 1,
      state: 'ok' as const,
      distance: 0.1,
      heardRatio: 1.0,
      margin: 0.7,
      vowelErrors: 1,
      vowelMargin: 0.04,
    },
    {
      surah: 1,
      ayah: 1,
      word: 2,
      state: 'skipped' as const,
      distance: 1.0,
      heardRatio: 0.1,
      margin: 0,
    },
    {
      surah: 1,
      ayah: 1,
      word: 3,
      state: 'ok' as const,
      distance: 0.08,
      heardRatio: 1.0,
      margin: 0.8,
    },
  ];

  const mistakes = evaluateVerdictMistakes(verdicts, words, 3);
  assert.equal(mistakes.length, 3);
  assert.equal(mistakes[0].word, 0);
  assert.equal(mistakes[0].kind, 'possible_substitution');
  assert.equal(mistakes[1].word, 1);
  assert.equal(mistakes[1].kind, 'possible_vowel');
  assert.equal(mistakes[2].word, 2);
  assert.equal(mistakes[2].kind, 'possible_omission');
});

test('@tilawi/quran-asr judgeAttemptTilawi aligns omissions, makhraj errors, and unattempted trailing words accurately', () => {
  const expected = ['بِسْمِ', 'ٱللَّهِ', 'ٱلرَّحْمَـٰنِ', 'ٱلرَّحِيمِ'];
  // User skips الله (word 1) and pronounces الرحمن as الرهمان (makhraj error on word 2), stops before word 3
  const heard = ['بسم', 'الرهمان'];
  const res = judgeAttemptTilawi(expected, heard);

  assert.equal(res.words[0].judgment, 'correct');
  assert.equal(res.words[1].operation, 'omission');
  assert.equal(res.words[1].judgment, 'apparent-error');
  assert.equal(res.words[2].operation, 'substitution');
  assert.equal(res.words[2].judgment, 'apparent-error');
  assert.ok(res.words[2].makhrajConfusion);
  assert.equal(res.words[3].operation, 'unattempted');
  assert.equal(res.words[3].judgment, 'unattempted');
  assert.equal(res.correct, 1);
  assert.equal(res.apparentErrors, 2);
});

test('@tilawi/quran-asr overlapMerge deduplicates overlapping transcript windows and getReciterAyahAudioUrl formats EveryAyah URL', () => {
  const merged = overlapMerge(['بسم', 'الله', 'الرحمن'], ['الله', 'الرحمن', 'الرحيم']);
  assert.deepEqual(merged, ['بسم', 'الله', 'الرحمن', 'الرحيم']);

  const url = getReciterAyahAudioUrl(1, 2, 'Husary_128kbps');
  assert.equal(url, 'https://everyayah.com/data/Husary_128kbps/001002.mp3');
});

