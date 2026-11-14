import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { computeRealRecitationScore, type ZipformerWordVerdict } from './mistakeDetector';
import {
  aggregateWeakAreas,
  appendSessionHistory,
  buildSessionRecord,
  formatTajweedFindingAz,
  getArticulationTipAz,
} from './sessionAnalytics';

const VERDICTS: ZipformerWordVerdict[] = [
  {
    surah: 1,
    ayah: 6,
    word: 0,
    state: 'ok',
    distance: 0,
    heardRatio: 1,
    margin: 1,
    gop: 0,
    expected: 'ءِهدِنَا',
    heard: 'ءِهدِنَا',
  },
  {
    surah: 1,
    ayah: 6,
    word: 1,
    state: 'wrong',
    distance: 0.28,
    heardRatio: 0.72,
    margin: 0.45,
    gop: -2.4,
    expected: 'صصِرَااطَ',
    heard: 'سسِرَااطَ',
  },
  {
    surah: 1,
    ayah: 6,
    word: 2,
    state: 'unsure',
    distance: 0.16,
    heardRatio: 0.88,
    margin: 0.68,
    gop: -1.1,
    vowelErrors: 1,
    expected: 'لمُستَقِيمَ',
    heard: 'لمُستَقِيمَ',
  },
];

const lookup = (s: number, a: number, w: number) => {
  const map: Record<string, string> = {
    '1:6:0': 'اهدنا',
    '1:6:1': 'الصراط',
    '1:6:2': 'المستقيم',
    '112:1:3': 'احد',
    '112:2:1': 'الصمد',
  };
  return map[`${s}:${a}:${w}`] ?? `w${w}`;
};

describe('sessionAnalytics', () => {
  it('getArticulationTipAz returns targeted Makhraj placement for letter swaps and Tajweed notes', () => {
    const sadSinTip = getArticulationTipAz({
      kind: 'possible_substitution',
      pairKey: 'ص/س',
    });
    assert.match(sadSinTip, /Sad.*qalın/i);

    const qafKafTip = getArticulationTipAz({
      kind: 'possible_substitution',
      pairKey: 'ق/ك',
    });
    assert.match(qafKafTip, /Qaf.*yumşaq damaq/i);

    const haTip = getArticulationTipAz({
      kind: 'possible_substitution',
      note: '«ح» (Ha) boğaz hərfi «ه» (Hə) kimi oxundu',
    });
    assert.match(haTip, /orta boğaz/i);

    const maddTip = getArticulationTipAz({
      kind: 'possible_vowel',
      note: 'Mədd uzadılması qısadır',
    });
    assert.match(maddTip, /2 hərəkə/i);

    const omitTip = getArticulationTipAz({ kind: 'possible_omission' });
    assert.match(omitTip, /nəfəsinizi/i);

    const repeatTip = getArticulationTipAz({ kind: 'possible_repetition' });
    assert.match(repeatTip, /təkrar/i);
  });

  it('formatTajweedFindingAz formats engine Tajweed rules cleanly in Azerbaijani', () => {
    const muttasil = formatTajweedFindingAz({
      s: 1,
      a: 7,
      w: 2,
      text: 'الضالين',
      type: 'muttasil',
      r: 1.1,
    });
    assert.equal(muttasil.ruleLabelAz, 'Məddi-Müttəsil (4–5 hərəkə)');
    assert.match(muttasil.detailAz, /həmzə/i);
    assert.equal(muttasil.isLengthShort, true);

    const qalqala = formatTajweedFindingAz({
      s: 112,
      a: 1,
      w: 3,
      text: 'أحد',
      kind: 'qalqala',
    });
    assert.equal(qalqala.ruleLabelAz, 'Qalqalə');

    const ghunna = formatTajweedFindingAz({
      s: 114,
      a: 1,
      w: 1,
      text: 'الناس',
      kind: 'ghunna',
      sub: 'ghunna',
    });
    assert.equal(ghunna.ruleLabelAz, 'Qunnə (Burun səsi)');
  });

  it('buildSessionRecord ignores tiny <2 word sessions and creates honest summary otherwise', () => {
    const tinyScore = computeRealRecitationScore(VERDICTS.slice(0, 1), 0, lookup);
    assert.equal(
      buildSessionRecord({
        page: 1,
        score: tinyScore,
        mistakes: [],
        timestamp: 1000,
      }),
      null
    );

    const realScore = computeRealRecitationScore(VERDICTS, 1, lookup);
    const rec = buildSessionRecord({
      page: 1,
      score: realScore,
      mistakes: [
        {
          s: 1,
          a: 6,
          w: 1,
          kind: 'possible_substitution',
          note: '«ص» (Sad) qalın hərfi «س» (Sin) kimi incə oxundu',
        },
      ],
      timestamp: 50_000,
    });
    assert.ok(rec);
    assert.equal(rec.page, 1);
    assert.equal(rec.evaluatedWords, 3);
    assert.ok(rec.weakTags.includes('Məxrəc ص/س'));
  });

  it('appendSessionHistory deduplicates same-page updates within 25s and aggregateWeakAreas ranks hotspots', () => {
    const score1 = computeRealRecitationScore(VERDICTS, 1, lookup);
    const rec1 = buildSessionRecord({
      page: 1,
      score: score1,
      mistakes: [
        {
          s: 1,
          a: 6,
          w: 1,
          kind: 'possible_substitution',
          note: '«ص» (Sad) qalın hərfi «س» (Sin) kimi incə oxundu',
        },
      ],
      timestamp: 100_000,
    })!;
    const rec2 = buildSessionRecord({
      page: 1,
      score: score1,
      mistakes: [
        {
          s: 1,
          a: 6,
          w: 1,
          kind: 'possible_substitution',
          note: '«ص» (Sad) qalın hərfi «س» (Sin) kimi incə oxundu',
        },
      ],
      timestamp: 112_000,
    })!;

    const score3 = computeRealRecitationScore(
      [
        {
          surah: 112,
          ayah: 1,
          word: 3,
          state: 'wrong',
          distance: 0.3,
          heardRatio: 0.7,
          margin: 0.4,
          gop: -2.2,
          expected: 'ءَحَد',
          heard: 'ءَهَد',
        },
        {
          surah: 112,
          ayah: 2,
          word: 1,
          state: 'wrong',
          distance: 0.29,
          heardRatio: 0.71,
          margin: 0.42,
          gop: -2.1,
          expected: 'صصَمَدُ',
          heard: 'سسَمَدُ',
        },
      ],
      2,
      lookup
    );
    const rec3 = buildSessionRecord({
      page: 604,
      score: score3,
      mistakes: [
        { s: 112, a: 1, w: 3, kind: 'possible_substitution', note: '«ح» (Ha) boğaz hərfi «ه» (Hə) kimi oxundu' },
        { s: 112, a: 2, w: 1, kind: 'possible_substitution', note: '«ص» (Sad) qalın hərfi «س» (Sin) kimi incə oxundu' },
      ],
      timestamp: 200_000,
    })!;

    let history = appendSessionHistory([], rec1);
    history = appendSessionHistory(history, rec2);
    assert.equal(history.length, 1);

    history = appendSessionHistory(history, rec3);
    assert.equal(history.length, 2);

    const { weakAreas, recurringSpots } = aggregateWeakAreas(history);
    assert.equal(weakAreas[0].tag, 'Məxrəc ص/س');
    assert.equal(weakAreas[0].count, 2);
    assert.match(weakAreas[0].tipAz, /Sad.*qalın/i);
    assert.ok(recurringSpots.length >= 2);
  });
});
