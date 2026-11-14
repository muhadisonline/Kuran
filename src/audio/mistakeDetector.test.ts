import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  computeRealRecitationScore,
  detectCompletedMistakesFromVerdicts,
  diagnosePerWordPhonemeMakhraj,
  diagnosePerWordSkeletonDiff,
  evaluateTajweedFindingsForMistakes,
  type ZipformerWordVerdict,
} from './mistakeDetector';

describe('diagnosePerWordPhonemeMakhraj', () => {
  it('detects subtle single-letter sister swaps even at very low edit distance', () => {
    const sadToSin = diagnosePerWordPhonemeMakhraj('صصِرَااطَ', 'سسِرَااطَ', {
      distance: 0.05,
      gop: -0.4,
      state: 'ok',
    });
    assert.ok(sadToSin);
    assert.equal(sadToSin.confusion.pairKey, 'ص/س');

    const qafToKaf = diagnosePerWordPhonemeMakhraj('لمُستَقِۦۦۦۦم', 'لمُستَكِۦۦم', {
      distance: 0.08,
      gop: -1.1,
      state: 'ok',
    });
    assert.ok(qafToKaf);
    assert.equal(qafToKaf.confusion.pairKey, 'ق/ك');

    const haToHeh = diagnosePerWordPhonemeMakhraj('ءَلحَمدُ', 'ءَلهَمدُ', {
      distance: 0.06,
      gop: -0.9,
      state: 'ok',
    });
    assert.ok(haToHeh);
    assert.equal(haToHeh.confusion.pairKey, 'ح/ه');
  });

  it('spares clean Qari elongation and superscript ya/waw quirks (0 false positives)', () => {
    const cleanElongations: Array<[string, string, number, number, ZipformerWordVerdict['state']]> = [
      ['ررَحِۦۦۦۦم', 'ررَحِۦۦم', 0.2, -1.7, 'ok'],
      ['ددِۦۦۦۦن', 'ددِۦۦن', 0.25, -1.6, 'unsure'],
      ['لعَاالَمِۦۦۦۦن', 'لعَاالَمِۦۦن', 0.14, -0.9, 'ok'],
      ['لمُستَقِۦۦۦۦم', 'لمُستَقِۦۦم', 0.15, -0.5, 'ok'],
      ['لمَغضُۥۥبِ', 'لمَغضُۥبِ', 0.12, -0.4, 'ok'],
    ];
    for (const [exp, hrd, distance, gop, state] of cleanElongations) {
      assert.equal(
        diagnosePerWordPhonemeMakhraj(exp, hrd, { distance, gop, state }),
        null,
        `Expected null for clean Qari pair ${exp} vs ${hrd}`
      );
    }
  });
});

describe('diagnosePerWordSkeletonDiff', () => {
  it('detects dropped interior Madd alif (مَـٰلِكِ -> مَلِكِ)', () => {
    const v: ZipformerWordVerdict = {
      surah: 1,
      ayah: 4,
      word: 0,
      state: 'unsure',
      distance: 0.14,
      heardRatio: 0.85,
      margin: 0.55,
      gop: -1.8,
      expected: 'مَاالِكِ',
      heard: 'مَلِكِ',
    };
    const res = diagnosePerWordSkeletonDiff(v, 'مالك');
    assert.ok(res);
    assert.match(res.titleAz, /Mədd/);
  });

  it('spares clean Qari when consonant skeletons match after normalization', () => {
    const v: ZipformerWordVerdict = {
      surah: 1,
      ayah: 4,
      word: 2,
      state: 'unsure',
      distance: 0.25,
      heardRatio: 0.78,
      margin: 0.99,
      gop: -1.6,
      expected: 'ددِۦۦۦۦن',
      heard: 'ددِۦۦن',
    };
    assert.equal(diagnosePerWordSkeletonDiff(v, 'الدين'), null);
  });
});

describe('detectCompletedMistakesFromVerdicts & telemetry breakdown', () => {
  const fatihaWords: Record<string, string> = {
    '1:4:0': 'ملك',
    '1:4:1': 'يوم',
    '1:4:2': 'الدين',
    '1:6:0': 'اهدنا',
    '1:6:1': 'الصراط',
    '1:6:2': 'المستقيم',
    '89:21:2': 'دكا',
    '89:21:3': 'دكا',
  };
  const lookup = (s: number, a: number, w: number) => fatihaWords[`${s}:${a}:${w}`] ?? `w${w}`;

  it('catches subtle Makhraj, composite GOP dead-zone (0.25), and multi-vowel errors while sparing Minshawy clean words', () => {
    const verdicts: ZipformerWordVerdict[] = [
      // Clean Minshawy-style unsure elongation on 1:4:2 -> MUST NOT false flag
      {
        surah: 1,
        ayah: 4,
        word: 2,
        state: 'unsure',
        distance: 0.25,
        heardRatio: 0.8,
        margin: 0.99,
        gop: -1.6,
        expected: 'ددِۦۦۦۦن',
        heard: 'ددِۦۦن',
      },
      // Subtle Makhraj ص -> س on 1:6:1 even with distance 0.06 & state ok
      {
        surah: 1,
        ayah: 6,
        word: 1,
        state: 'ok',
        distance: 0.06,
        heardRatio: 0.95,
        margin: 0.82,
        gop: -0.5,
        expected: 'صصِرَااطَ',
        heard: 'سسِرَااطَ',
      },
      // Dead-zone skeleton diff substitution on 1:6:0 (distance 0.24, unsure, gop -3.7, consonant distortion)
      {
        surah: 1,
        ayah: 6,
        word: 0,
        state: 'unsure',
        distance: 0.24,
        heardRatio: 0.72,
        margin: 0.52,
        gop: -3.7,
        expected: 'ءِهدِنَ',
        heard: 'ءِهدِمَ',
      },
      // Composite GOP dead-zone fallback when skeleton LCS >= 0.85 (e.g. 10-letter word with gop -3.7)
      {
        surah: 1,
        ayah: 5,
        word: 1,
        state: 'unsure',
        distance: 0.25,
        heardRatio: 0.72,
        margin: 0.50,
        gop: -3.7,
        expected: 'لمُستَغفِرِۦۦنَ',
        heard: 'لمُستَكفِرِۦۦنَ',
      },
      // Multi-vowel Harakat error on 1:4:1
      {
        surah: 1,
        ayah: 4,
        word: 1,
        state: 'ok',
        distance: 0.18,
        heardRatio: 0.92,
        margin: 0.65,
        gop: -1.4,
        vowelErrors: 2,
        vowelMargin: 0.036,
        expected: 'يَومِ',
        heard: 'يَومِ',
      },
      // Clean waqf ending vowel dropped on 1:4:0 (should NOT false flag)
      {
        surah: 1,
        ayah: 4,
        word: 0,
        state: 'ok',
        distance: 0.04,
        heardRatio: 0.95,
        margin: 0.92,
        gop: -0.3,
        expected: 'نَعبُدُ',
        heard: 'نَعبُد',
      },
      // Sentinel clean next word
      {
        surah: 1,
        ayah: 6,
        word: 2,
        state: 'ok',
        distance: 0.0,
        heardRatio: 1.0,
        margin: 1.0,
        gop: 0,
        expected: 'لمُستَقِۦۦۦۦم',
        heard: 'لمُستَقِۦۦۦۦم',
      },
    ];

    const detected = detectCompletedMistakesFromVerdicts(verdicts, lookup, { isSessionEnding: true });
    const byKey = new Map(detected.map((m) => [`${m.surah}:${m.ayah}:${m.word}`, m]));

    assert.equal(byKey.has('1:4:2'), false, 'Clean Qari elongation must NOT be flagged');
    assert.equal(byKey.has('1:4:0'), false, 'Clean waqf final-vowel pause must NOT be flagged');
    assert.equal(byKey.get('1:6:1')?.subKind, 'makhraj');
    assert.equal(byKey.get('1:6:0')?.subKind, 'substitution');
    assert.equal(byKey.get('1:5:1')?.subKind, 'gop');
    assert.equal(byKey.get('1:4:1')?.kind, 'possible_vowel');

    // Interior Harakat swap via phoneme comparison (e.g. رَببِ -> رُببِ)
    const interiorVowel = detectCompletedMistakesFromVerdicts(
      [
        {
          surah: 1,
          ayah: 2,
          word: 2,
          state: 'ok',
          distance: 0.12,
          heardRatio: 0.95,
          margin: 0.78,
          gop: -1.3,
          expected: 'رَببِ',
          heard: 'رُببِ',
        },
        {
          surah: 1,
          ayah: 2,
          word: 3,
          state: 'ok',
          distance: 0,
          heardRatio: 1,
          margin: 1,
          gop: 0,
          expected: 'لعَاالَمِۦۦۦۦن',
          heard: 'لعَاالَمِۦۦۦۦن',
        },
      ],
      lookup,
      { isSessionEnding: true }
    );
    assert.equal(interiorVowel.length, 1);
    assert.equal(interiorVowel[0].kind, 'possible_vowel');

    const score = computeRealRecitationScore(verdicts, detected.length, lookup);
    const makhrajRow = score.breakdowns.find((b) => b.surah === 1 && b.ayah === 6 && b.word === 1);
    assert.equal(makhrajRow?.reasonTag, 'Məxrəc');
    assert.ok((makhrajRow?.wordScorePercent ?? 100) <= 48);

    const cleanWithHighSlip = computeRealRecitationScore(
      [
        {
          surah: 1,
          ayah: 5,
          word: 1,
          state: 'ok',
          distance: 0,
          heardRatio: 1,
          margin: 1,
          gop: 0,
          slip: 0.999,
          expected: 'نَعبُدُ',
          heard: 'نَعبُدُ',
        },
      ],
      0,
      lookup
    );
    assert.equal(cleanWithHighSlip.cleanWords, 1);
    assert.equal(cleanWithHighSlip.breakdowns[0].reasonTag, undefined);
    assert.equal(cleanWithHighSlip.breakdowns[0].wordScorePercent, 100);
  });

  it('spares Quranic consecutive doubled words like دكا دكا unless extreme repetition spike', () => {
    const verdicts: ZipformerWordVerdict[] = [
      {
        surah: 89,
        ayah: 21,
        word: 2,
        state: 'ok',
        distance: 0,
        heardRatio: 1,
        margin: 1,
        repGain: 5.1,
        expected: 'دَککَ',
        heard: 'دَککَ',
      },
      {
        surah: 89,
        ayah: 21,
        word: 3,
        state: 'ok',
        distance: 0,
        heardRatio: 1,
        margin: 1,
        expected: 'دَککَ',
        heard: 'دَککَ',
      },
    ];
    const detected = detectCompletedMistakesFromVerdicts(verdicts, lookup, { isSessionEnding: true });
    assert.equal(detected.length, 0);
  });

  it('spares clean Qari connected-speech boundary bleed, truncated slices, glide insertions, and word-0 cold start (0 false positives)', () => {
    const cleanBenchLookup = (s: number, a: number, w: number) => {
      const map: Record<string, string[]> = {
        '1:1': ['بسم', 'الله', 'الرحمن', 'الرحيم'],
        '1:4': ['ملك', 'يوم', 'الدين'],
        '1:5': ['إياك', 'نعبد', 'وإياك', 'نستعين'],
        '1:7': ['صراط', 'الذين', 'أنعمت', 'عليهم', 'غير', 'المغضوب', 'عليهم', 'ولا', 'الضالين'],
        '112:1': ['قل', 'هو', 'الله', 'أحد'],
        '114:5': ['الذي', 'يوسوس', 'في', 'صدور', 'الناس'],
      };
      return map[`${s}:${a}`]?.[w] ?? `w${w}`;
    };

    const qariVerdicts: ZipformerWordVerdict[] = [
      // Word 0 mic cold start onset trim: بِسمِ -> سمِ
      { surah: 1, ayah: 1, word: 0, state: 'unsure', distance: 0.25, heardRatio: 0.72, margin: 0.5, gop: -0.69, expected: 'بِسمِ', heard: 'سمِ' },
      // Soft final 'ه' pause on state ok: للَااه -> للَاا
      { surah: 1, ayah: 1, word: 1, state: 'ok', distance: 0.17, heardRatio: 0.90, margin: 0.65, gop: -0.5, expected: 'للَااهِ', heard: 'للَا' },
      // Boundary bleed from الرحيم onto الرحمن: للَحمَاانِمِ
      { surah: 1, ayah: 1, word: 2, state: 'ok', distance: 0.17, heardRatio: 0.92, margin: 0.88, gop: -0.11, expected: 'للَحمَاانِ', heard: 'للَحمَاانِمِ' },
      { surah: 1, ayah: 1, word: 3, state: 'ok', distance: 0.02, heardRatio: 1.0, margin: 0.98, gop: -0.1, expected: 'ررَحِۦۦم', heard: 'ررَحِۦۦم' },
      // Truncated CTC fragment مَ on مَاالِكِ with clean GOP
      { surah: 1, ayah: 4, word: 0, state: 'wrong', distance: 0.65, heardRatio: 0.72, margin: 0.60, gop: -0.2, expected: 'مَاالِكِ', heard: 'مَ' },
      // Boundary bleed يوم -> يوممدد before الدين
      { surah: 1, ayah: 4, word: 1, state: 'wrong', distance: 0.48, heardRatio: 0.85, margin: 0.52, gop: -1.1, expected: 'يَومِ', heard: 'يَومِمِددِ' },
      { surah: 1, ayah: 4, word: 2, state: 'ok', distance: 0.03, heardRatio: 1.0, margin: 0.95, gop: -0.2, expected: 'ددِۦۦن', heard: 'ددِۦۦن' },
      // Connected bleed إياك -> إياكت before نعبد on clean ok
      { surah: 1, ayah: 5, word: 0, state: 'ok', distance: 0.16, heardRatio: 0.94, margin: 0.82, gop: -0.45, expected: 'ءِييَااكَ', heard: 'ءِييَااكَتتَ' },
      { surah: 1, ayah: 5, word: 1, state: 'ok', distance: 0.0, heardRatio: 1.0, margin: 0.99, gop: -0.1, expected: 'نَعبُدُ', heard: 'نَعبُدُ' },
      // Connected bleed ولا -> ولض before الضالين
      { surah: 1, ayah: 7, word: 7, state: 'wrong', distance: 0.42, heardRatio: 0.88, margin: 0.58, gop: -0.9, expected: 'وَلَا', heard: 'وَلَضضَ' },
      { surah: 1, ayah: 7, word: 8, state: 'ok', distance: 0.04, heardRatio: 0.98, margin: 0.95, gop: -0.3, expected: 'ضضَاالِۦۦن', heard: 'ضضَاالِۦۦن' },
      // Truncated CTC fragment يُ on يُوَسْوِسُ & glide insertion صُۥۥدُۥۥرِ -> ودور on ok
      { surah: 114, ayah: 5, word: 1, state: 'unsure', distance: 0.55, heardRatio: 0.68, margin: 0.45, gop: -1.5, expected: 'يُوَسوِسُ', heard: 'يُ' },
      { surah: 114, ayah: 5, word: 3, state: 'ok', distance: 0.20, heardRatio: 0.96, margin: 0.72, gop: -1.52, expected: 'صُدُۥۥرِ', heard: 'صُوۥدُۥۥرِ' },
      { surah: 114, ayah: 5, word: 4, state: 'ok', distance: 0.0, heardRatio: 1.0, margin: 0.99, gop: -0.1, expected: 'ننَااس', heard: 'ننَااس' },
    ];

    const falseFlags = detectCompletedMistakesFromVerdicts(qariVerdicts, cleanBenchLookup, {
      isSessionEnding: true,
    });
    assert.deepEqual(falseFlags, []);
  });

  it('catches ذ/ظ, guarded dental pairs, composite صِرَٰطَ->سرط, neighbor-anchored omissions, and Tajweed LLR/Madd violations', () => {
    const customLookup = (_s: number, _a: number, w: number) =>
      ['الذين', 'صراط', 'نستعين', 'مالك', 'الضالين'][w] ?? `w${w}`;

    // 1. ذ -> ظ swap + composite صِرَٰطَ -> سرط + guarded ث -> ت with real acoustic drop + neighbor omission (heardRatio 0.42)
    const dhalToZha = diagnosePerWordPhonemeMakhraj('لذِۦۦنَ', 'لظِۦۦنَ', {
      distance: 0.10,
      gop: -1.4,
      state: 'unsure',
    });
    assert.equal(dhalToZha?.confusion.pairKey, 'ذ/ظ');

    const compositeSadAlef = diagnosePerWordPhonemeMakhraj('صصِرَااطَ', 'سسِرَطَ', {
      distance: 0.18,
      gop: -1.8,
      state: 'unsure',
    });
    assert.equal(compositeSadAlef?.confusion.pairKey, 'ص/س');

    // Guarded dental ث -> ت: spared when ultra-clean low distance on ok, caught when real distortion
    assert.equal(
      diagnosePerWordPhonemeMakhraj('ثُممَ', 'تُممَ', { distance: 0.03, gop: -0.4, state: 'ok' }),
      null
    );
    assert.equal(
      diagnosePerWordPhonemeMakhraj('ثُممَ', 'تُممَ', { distance: 0.14, gop: -1.6, state: 'unsure' })?.confusion.pairKey,
      'ث/ت'
    );

    const verdicts: ZipformerWordVerdict[] = [
      { surah: 1, ayah: 7, word: 0, state: 'ok', distance: 0.02, heardRatio: 1.0, margin: 0.96, gop: -0.2, expected: 'صصِرَااطَ', heard: 'صصِرَااطَ' },
      // Neighbor-anchored rushed omission (heardRatio: 0.42 between two ok words)
      { surah: 1, ayah: 7, word: 1, state: 'skipped', distance: 0.85, heardRatio: 0.42, margin: 0.1, gop: -6.5 },
      { surah: 1, ayah: 7, word: 2, state: 'ok', distance: 0.01, heardRatio: 1.0, margin: 0.98, gop: -0.1, expected: 'ءَنعَمتَ', heard: 'ءَنعَمتَ' },
    ];
    const detected = detectCompletedMistakesFromVerdicts(verdicts, customLookup, { isSessionEnding: true });
    assert.equal(detected.length, 1);
    assert.equal(detected[0].kind, 'possible_omission');
    assert.equal(detected[0].word, 1);

    // Tajweed engine event promotion: catches severe clipped Madd 6 + LLR letter mismatch, spares ok natural madd
    const tjMistakes = evaluateTajweedFindingsForMistakes(
      [
        { s: 1, a: 7, w: 4, kind: 'madd', sub: 'madd6', type: 'long', r: 1.6, rRef: 4.5, text: 'الضالين' },
        { s: 1, a: 7, w: 0, kind: 'letter', sub: 'ص/س', expected: 'ص', heard: 'س', text: 'صراط' },
        // Natural madd on a clean ok word must NOT false flag
        { s: 1, a: 4, w: 0, kind: 'madd', sub: 'madd2', type: 'tabii', r: 0.8, rRef: 1.8, text: 'مالك' },
      ],
      customLookup,
      [
        { surah: 1, ayah: 7, word: 4, state: 'unsure', distance: 0.15, heardRatio: 0.65, margin: 0.6, gop: -1.4 },
        { surah: 1, ayah: 7, word: 0, state: 'unsure', distance: 0.14, heardRatio: 0.9, margin: 0.55, gop: -1.8 },
        { surah: 1, ayah: 4, word: 0, state: 'ok', distance: 0.02, heardRatio: 1.0, margin: 0.95, gop: -0.2 },
      ]
    );
    assert.equal(tjMistakes.length, 2);
    assert.equal(tjMistakes[0].word, 4);
    assert.equal(tjMistakes[0].kind, 'possible_vowel');
    assert.equal(tjMistakes[1].word, 0);
    assert.equal(tjMistakes[1].subKind, 'makhraj');
  });

  it('spares noisy-mic single-vowel fluctuation, clean elongation unsure states, and brisk clean-GOP function words while catching GOP-backed errors', () => {
    const baqarahLookup = (s: number, a: number, w: number) => {
      if (s === 2 && a === 7) return ['ختم', 'الله', 'على', 'قلوبهم', 'عظيم'][w] ?? `w${w}`;
      return ['في', 'قلوبهم', 'مرض'][w] ?? `w${w}`;
    };

    const noisyHumanVerdicts: ZipformerWordVerdict[] = [
      // Brisk function word 'في' with clean GOP (-1.0) must NOT trigger omission
      { surah: 2, ayah: 10, word: 0, state: 'skipped', distance: 0.42, heardRatio: 0.41, margin: 0.4, gop: -1.0 },
      // Noisy single-vowel fluctuation with clean GOP (-0.6) & margin 0.048 must NOT false flag
      {
        surah: 2,
        ayah: 10,
        word: 1,
        state: 'ok',
        distance: 0.08,
        heardRatio: 0.94,
        margin: 0.72,
        gop: -0.6,
        vowelErrors: 1,
        vowelMargin: 0.048,
        expected: 'قُلُۥۥبِهِم',
        heard: 'قُلُۥۥبِهِم',
      },
      // Clean elongation difference عَظِۦۦۦۦم -> عَظِۦۦم even when labeled unsure by aligner
      {
        surah: 2,
        ayah: 7,
        word: 4,
        state: 'unsure',
        distance: 0.22,
        heardRatio: 0.78,
        margin: 0.9,
        gop: -2.5,
        vowelErrors: 0,
        expected: 'عَظِۦۦۦۦم',
        heard: 'عَظِۦۦم',
      },
      // Genuine GOP-backed single-vowel swap -> MUST catch!
      {
        surah: 2,
        ayah: 10,
        word: 2,
        state: 'unsure',
        distance: 0.14,
        heardRatio: 0.92,
        margin: 0.7,
        gop: -2.1,
        vowelErrors: 1,
        vowelMargin: 0.056,
        expected: 'مَرَض',
        heard: 'مَرَض',
      },
    ];

    const detected = detectCompletedMistakesFromVerdicts(noisyHumanVerdicts, baqarahLookup, {
      isSessionEnding: true,
    });
    assert.equal(detected.length, 1);
    assert.equal(detected[0].word, 2);
    assert.equal(detected[0].kind, 'possible_vowel');

    const score = computeRealRecitationScore(
      [noisyHumanVerdicts[1], noisyHumanVerdicts[2]],
      0,
      baqarahLookup
    );
    assert.equal(score.cleanWords, 2);
    assert.equal(score.unsureWords, 0);
    assert.equal(score.errorWords, 0);
  });
});

