/**
 * Actionable Makhraj & Tajweed Pedagogical Guides + Honest Session Progress Analytics.
 * Stores real user recitation sessions & recurring weak articulation spots (zero fake stats).
 */

import {
  MistakeKind,
  RealRecitationScore,
  detectMakhrajConfusion,
  diagnosePerWordPhonemeMakhraj,
} from './mistakeDetector';

export const RECITATION_HISTORY_KEY = 'tartil.history.v1';

export interface MakhrajArticulationGuide {
  titleAz: string;
  tipAz: string;
}

const MAKHRAJ_GUIDE_BY_LETTER: Record<string, string> = {
  ص: '«ص» (Sad) qalın hərfdir: dil ucunu alt ön dişlərin arxasına yaxınlaşdırıb, dilin arxasını yumşaq damağa doğru qaldıraraq qalın «s» kimi tələffüz edin.',
  س: '«س» (Sin) incə hərfdir: dilin arxasını qaldırmadan, dil ucunu alt dişlərin arxasına yönəldərək yüngül və incə çıxarın.',
  ح: '«ح» (Ha) orta boğazdan çıxır: boğazı hafif sıxaraq təmiz və xırıltısız isti hava verin («ه» kimi sinədən yumşaq buraxmayın).',
  ه: '«ه» (Hə) ən alt boğazdan (sinəyə yaxın) gəlir: boğazı sıxmadan sərbəst nəfəslə tələffüz edin.',
  ع: '«ع» (Ayn) orta boğaz əzələsinin yığılması ilə çıxır: «ء/ا» kimi kəskin kəsilmiş həmzəyə çevirməyin.',
  ء: '«ء» (Həmzə) səs tellərinin ani qapanıb-açılması ilə boğazın ən altından aydın çıxarılır.',
  ا: '«ا» (Əlif) boğazı sıxmadan açıq ağız boşluğundan təmiz səsləndirilir.',
  ض: '«ض» (Dad) dilin bir və ya hər iki yan kənarını üst azı dişlərin iç tərəfinə söykəyib dilin arxasını qaldıraraq qalın çıxarılır («د» və ya «ظ» kimi oxumayın).',
  د: '«د» (Dəl) incədir: dil ucunu üst ön dişlərin ətinə vuraraq qalınlaşdırmadan oxuyun.',
  ظ: '«ظ» (Za) dil ucunu üst kəsici dişlərin ucuna toxundurub dilin arxasını qaldıraraq qalın-pəltək tələffüz edin.',
  ط: '«ط» (Ta) qalın hərfdir: dil ucu üst ön dişlərin dibinə dəyərkən dilin arxası da yuxarı qalxmalıdır; nəfəs axını buraxmayın.',
  ت: '«ت» (Tə) incə hərfdir: dilin arxasını qaldırmadan üst diş ətindən yüngülcə çıxarın.',
  ق: '«ق» (Qaf) ən arxa yumşaq damaqdan qalın və qalqaləyə meylli çıxarılır; «ك» kimi önə gətirməyin.',
  ك: '«ك» (Kəf) sərt damaqla yumşaq damağın sərhədindən (bir az daha öndən) incə çıxarılır.',
  ذ: '«ذ» (Zəl) pəltəkdir: dil ucunu üst kəsici dişlərin kənarına toxundurun; «ز» kimi vızıldamayın.',
  ز: '«ز» (Zə) dil ucunu alt dişlərin arxasına yaxın tutaraq kəskin və təmiz çıxarın.',
  ث: '«ث» (Sə) pəltəkdir: dil ucunu üst ön dişlərin kənarına yumşaq toxundurub hava verin; «س» kimi oxumayın.',
  خ: '«خ» (Xa) yuxarı boğazdan (dil kökünə yaxın) yüngül xırıltı ilə qalın tələffüz olunur.',
  غ: '«غ» (Ğayn) yuxarı boğazdan səsli və qalın çıxır (xırıltılı «خ» ilə qarışdırmayın).',
};

/**
 * Returns a practical Azerbaijani pronunciation & tongue/throat placement tip
 * for the detected mistake category or Makhraj confusion pair.
 */
export function getArticulationTipAz(params: {
  kind: MistakeKind;
  expectedWord?: string;
  heardWord?: string;
  pairKey?: string;
  note?: string;
}): string {
  const { kind, expectedWord, heardWord, pairKey, note } = params;

  if (pairKey) {
    const [expChar] = pairKey.split('/');
    if (expChar && MAKHRAJ_GUIDE_BY_LETTER[expChar]) {
      return MAKHRAJ_GUIDE_BY_LETTER[expChar];
    }
  }

  if (expectedWord && heardWord) {
    const perWord = diagnosePerWordPhonemeMakhraj(expectedWord, heardWord, { distance: 0.12 });
    if (perWord && MAKHRAJ_GUIDE_BY_LETTER[perWord.confusion.expectedLetter]) {
      return MAKHRAJ_GUIDE_BY_LETTER[perWord.confusion.expectedLetter];
    }
    const textMakhraj = detectMakhrajConfusion(expectedWord, heardWord);
    if (textMakhraj && MAKHRAJ_GUIDE_BY_LETTER[textMakhraj.expectedLetter]) {
      return MAKHRAJ_GUIDE_BY_LETTER[textMakhraj.expectedLetter];
    }
  }

  if (note) {
    const letterMatch = note.match(/«([صسحعهءادضظطتقكذزثخغ])»/);
    if (letterMatch && MAKHRAJ_GUIDE_BY_LETTER[letterMatch[1]]) {
      return MAKHRAJ_GUIDE_BY_LETTER[letterMatch[1]];
    }
    if (note.includes('Mədd') || note.includes('uzatma')) {
      return 'Əlif (ا), Vav (و) və Yə (ي) mədlərini ən azı 2 hərəkə (təbii mədd) miqdarında bərabər uzadın.';
    }
  }

  switch (kind) {
    case 'possible_vowel':
      return 'Qısa saitləri — Fəthə (a/ə), Kəsrə (i), Zəmmə (u) — qonşu hərflərə qarışdırmadan aydın tələffüz edin; yalnız ayə sonunda dayanarkən sükun verin.';
    case 'possible_omission':
    case 'possible_skipped_ayah':
      return 'Sözlər arasındakı keçiddə tələsməyin; nəfəsinizi ayənin məna hissələrinə görə bölüb hər sözü tam səsləndirin.';
    case 'possible_repetition':
      return 'Şəddəli və ya oxşar hecaları yalnız yüngül vurğu ilə saxlayın; bütöv sözü iki dəfə təkrar etməyin.';
    default:
      return 'Hərfin çıxış yerini (məxrəcini) və uzun saitləri yavaş sürətdə ayırıb «Qarini dinlə» ilə müqayisəli məşq edin.';
  }
}

export interface RawEngineTajweedItem {
  s: number;
  a: number;
  w: number;
  text?: string;
  kind?: string;
  sub?: string;
  type?: string;
  r?: number;
  rRef?: number | null;
  expected?: string;
  heard?: string;
}

export interface TajweedFindingAz {
  key: string;
  surah: number;
  ayah: number;
  word: number;
  wordText: string;
  ruleLabelAz: string;
  detailAz: string;
  isLengthShort: boolean;
}

/**
 * Translates raw Zipformer acoustic Tajweed engine findings (`window.__tj.found`)
 * into clear Azerbaijani guidance without touching or altering engine thresholds.
 */
export function formatTajweedFindingAz(item: RawEngineTajweedItem): TajweedFindingAz {
  const s = Number(item.s || 0);
  const a = Number(item.a || 0);
  const w = Number(item.w ?? 0);
  const key = `${s}:${a}:${w}:${item.kind || 'rule'}`;
  const kind = item.kind || '';
  const sub = item.sub || '';
  const type = item.type || '';
  const ratioShort = typeof item.r === 'number' && item.r < 1.35;

  let ruleLabelAz = 'Təcvid qaydası';
  let detailAz = 'Təcvid ölçüsünü yoxlayın';

  if (type === 'long' || sub === 'madd6') {
    ruleLabelAz = 'Məddi-Lazım (6 hərəkə)';
    detailAz = 'Bu mədd 6 hərəkə tam uzadılmalıdır (qısa oxundu).';
  } else if (type === 'munfasil') {
    ruleLabelAz = 'Məddi-Münfəsil (4–5 hərəkə)';
    detailAz = 'Məddi-Münfəsil 4–5 hərəkə uzadılmalıdır.';
  } else if (type === 'muttasil') {
    ruleLabelAz = 'Məddi-Müttəsil (4–5 hərəkə)';
    detailAz = 'Eyni sözdəki həmzəyə görə 4–5 hərəkə uzadılmalıdır.';
  } else if (kind === 'madd') {
    ruleLabelAz = 'Təbii Mədd (2 hərəkə)';
    detailAz = ratioShort ? 'Uzatma çox qısa kəsildi (minimum 2 hərəkə olmalıdır).' : 'Uzatma müddəti yetərli olmadı.';
  } else if (kind === 'ghunna' || sub === 'ghunna' || sub === 'extra-ghunna') {
    ruleLabelAz = 'Qunnə (Burun səsi)';
    detailAz =
      sub === 'extra-ghunna'
        ? 'Burada burun səsi (qunnə) artıq saxlandı.'
        : 'Şəddəli نّ / مّ hərflərində burun səsi 2 hərəkə saxlanmalıdır.';
  } else if (kind === 'ikhfa' || sub === 'ikhfa') {
    ruleLabelAz = 'İxfə / İdğam';
    detailAz = 'Sakin nun və ya tənvin gizlədilərək (qunnə ilə) oxunmalıdır.';
  } else if (kind === 'qalqala' || sub === 'qalqala' || sub === 'qalqala-mid') {
    ruleLabelAz = 'Qalqalə';
    detailAz = 'Sakin ق ط ب ج د hərflərində səs yumşaq sıçrayışla vurğulanmalıdır.';
  } else if (kind === 'waqf') {
    ruleLabelAz = sub === 'waqf-soft' ? 'Yumşaq Vəqf' : 'Vəqf (Dayanma)';
    detailAz = 'Söz sonundakı dayanma/vəqf səsini aydın saxlayın.';
  } else if (kind === 'letter') {
    ruleLabelAz = 'Hərf təcvidi';
    detailAz = 'Təcvidli hərf çıxışında fərq eşidildi.';
  }

  return {
    key,
    surah: s,
    ayah: a,
    word: w,
    wordText: item.text || `Söz ${w + 1}`,
    ruleLabelAz,
    detailAz,
    isLengthShort: ratioShort,
  };
}

export interface SessionWeakSpotRecord {
  key: string; // surah:ayah:word
  surah: number;
  ayah: number;
  word: number;
  wordText: string;
  tag: string;
}

export interface RecitationSessionRecord {
  id: string;
  timestamp: number;
  page: number;
  surah: number;
  ayahFrom: number;
  ayahTo: number;
  evaluatedWords: number;
  scorePercent: number;
  errorWords: number;
  weakTags: string[];
  weakSpots: SessionWeakSpotRecord[];
}

export interface AggregatedWeakArea {
  tag: string;
  count: number;
  lastSeen: number;
  tipAz: string;
}

export function extractWeakTagsForSession(
  score: RealRecitationScore,
  mistakes: Array<{ s: number; a: number; w: number; kind: MistakeKind; note?: string }>
): { weakTags: string[]; weakSpots: SessionWeakSpotRecord[] } {
  const tags: string[] = [];
  const spots: SessionWeakSpotRecord[] = [];
  const seenSpotKeys = new Set<string>();

  for (const m of mistakes) {
    const locKey = `${m.s}:${m.a}:${m.w}`;
    const b = score.breakdowns.find((x) => x.surah === m.s && x.ayah === m.a && x.word === m.w);
    const wordText = b?.wordText || `Söz ${m.w + 1}`;
    let tag = 'Tələffüz';

    if (m.note) {
      const makhrajMatch = m.note.match(/«([صسحعهءادضظطتقكذزثخغ])»[\s\S]*?«([صسحعهءادضظطتقكذزثخغ])»/);
      if (makhrajMatch) {
        tag = `Məxrəc ${makhrajMatch[1]}/${makhrajMatch[2]}`;
      } else if (m.note.includes('Mədd')) {
        tag = 'Mədd uzatması';
      } else if (m.kind === 'possible_vowel') {
        tag = 'Hərəkə';
      } else if (m.kind === 'possible_omission' || m.kind === 'possible_skipped_ayah') {
        tag = 'Söz buraxılması';
      } else if (m.kind === 'possible_repetition') {
        tag = 'Təkrar';
      } else if (b?.reasonTag === 'GOP') {
        tag = 'Tələffüz / GOP';
      }
    } else if (m.kind === 'possible_vowel') {
      tag = 'Hərəkə';
    } else if (m.kind === 'possible_omission') {
      tag = 'Söz buraxılması';
    } else if (m.kind === 'possible_repetition') {
      tag = 'Təkrar';
    }

    tags.push(tag);
    if (!seenSpotKeys.has(locKey)) {
      seenSpotKeys.add(locKey);
      spots.push({
        key: locKey,
        surah: m.s,
        ayah: m.a,
        word: m.w,
        wordText,
        tag,
      });
    }
  }

  for (const b of score.breakdowns) {
    if (b.reasonTag === 'Məxrəc' || b.reasonTag === 'Hərəkə' || b.reasonTag === 'GOP') {
      const locKey = `${b.surah}:${b.ayah}:${b.word}`;
      if (!seenSpotKeys.has(locKey)) {
        seenSpotKeys.add(locKey);
        tags.push(b.reasonTag === 'GOP' ? 'Tələffüz / GOP' : b.reasonTag);
        spots.push({
          key: locKey,
          surah: b.surah,
          ayah: b.ayah,
          word: b.word,
          wordText: b.wordText,
          tag: b.reasonTag,
        });
      }
    }
  }

  return { weakTags: tags, weakSpots: spots };
}

export function buildSessionRecord(params: {
  page: number;
  score: RealRecitationScore;
  mistakes: Array<{ s: number; a: number; w: number; kind: MistakeKind; note?: string }>;
  timestamp?: number;
}): RecitationSessionRecord | null {
  const { page, score, mistakes, timestamp = Date.now() } = params;
  if (!score.hasData || score.scorePercent === null || score.evaluatedWords < 2) {
    return null;
  }
  const surahs = score.breakdowns.map((b) => b.surah).filter((s) => s > 0);
  const ayahs = score.breakdowns.map((b) => b.ayah).filter((a) => a > 0);
  const surah = surahs[0] ?? 1;
  const ayahFrom = ayahs.length > 0 ? Math.min(...ayahs) : 1;
  const ayahTo = ayahs.length > 0 ? Math.max(...ayahs) : ayahFrom;
  const { weakTags, weakSpots } = extractWeakTagsForSession(score, mistakes);

  return {
    id: `${timestamp}-${page}-${score.evaluatedWords}`,
    timestamp,
    page,
    surah,
    ayahFrom,
    ayahTo,
    evaluatedWords: score.evaluatedWords,
    scorePercent: score.scorePercent,
    errorWords: score.errorWords,
    weakTags,
    weakSpots,
  };
}

export function appendSessionHistory(
  existing: RecitationSessionRecord[],
  next: RecitationSessionRecord | null,
  maxItems: number = 30
): RecitationSessionRecord[] {
  if (!next) return existing;
  const latest = existing[0];
  if (
    latest &&
    latest.page === next.page &&
    latest.evaluatedWords === next.evaluatedWords &&
    latest.scorePercent === next.scorePercent &&
    Math.abs(next.timestamp - latest.timestamp) < 25000
  ) {
    return existing;
  }
  return [next, ...existing].slice(0, maxItems);
}

export function aggregateWeakAreas(sessions: RecitationSessionRecord[]): {
  weakAreas: AggregatedWeakArea[];
  recurringSpots: Array<{ key: string; surah: number; ayah: number; word: number; wordText: string; tag: string; count: number }>;
} {
  const tagCounts = new Map<string, { count: number; lastSeen: number }>();
  const spotCounts = new Map<
    string,
    { surah: number; ayah: number; word: number; wordText: string; tag: string; count: number }
  >();

  for (const s of sessions) {
    for (const tag of s.weakTags) {
      const prev = tagCounts.get(tag) ?? { count: 0, lastSeen: 0 };
      tagCounts.set(tag, {
        count: prev.count + 1,
        lastSeen: Math.max(prev.lastSeen, s.timestamp),
      });
    }
    for (const sp of s.weakSpots) {
      const prev = spotCounts.get(sp.key);
      if (prev) {
        prev.count += 1;
      } else {
        spotCounts.set(sp.key, {
          surah: sp.surah,
          ayah: sp.ayah,
          word: sp.word,
          wordText: sp.wordText,
          tag: sp.tag,
          count: 1,
        });
      }
    }
  }

  const weakAreas: AggregatedWeakArea[] = Array.from(tagCounts.entries())
    .map(([tag, meta]) => {
      let tipAz = 'Sözləri aramla oxuyub «Qarini dinlə» ilə müqayisə edin.';
      const m = tag.match(/^Məxrəc ([صسحعهءادضظطتقكذزثخغ])/);
      if (m && MAKHRAJ_GUIDE_BY_LETTER[m[1]]) {
        tipAz = MAKHRAJ_GUIDE_BY_LETTER[m[1]];
      } else if (tag === 'Hərəkə') {
        tipAz = getArticulationTipAz({ kind: 'possible_vowel' });
      } else if (tag === 'Mədd uzatması') {
        tipAz = 'Əlif, Vav və Yə mədlərini ən azı 2 hərəkə uzadın.';
      } else if (tag === 'Söz buraxılması') {
        tipAz = getArticulationTipAz({ kind: 'possible_omission' });
      } else if (tag === 'Təkrar') {
        tipAz = getArticulationTipAz({ kind: 'possible_repetition' });
      }
      return { tag, count: meta.count, lastSeen: meta.lastSeen, tipAz };
    })
    .sort((a, b) => b.count - a.count || b.lastSeen - a.lastSeen)
    .slice(0, 6);

  const recurringSpots = Array.from(spotCounts.entries())
    .map(([key, val]) => ({ key, ...val }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 8);

  return { weakAreas, recurringSpots };
}

export function loadRecitationHistory(): RecitationSessionRecord[] {
  try {
    const raw = localStorage.getItem(RECITATION_HISTORY_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function saveRecitationHistory(records: RecitationSessionRecord[]): void {
  try {
    localStorage.setItem(RECITATION_HISTORY_KEY, JSON.stringify(records));
  } catch {
    // ignore storage quota errors
  }
}

export function clearRecitationHistoryStorage(): void {
  try {
    localStorage.removeItem(RECITATION_HISTORY_KEY);
  } catch {
    // ignore
  }
}
