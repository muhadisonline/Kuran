import {
  VOICE_BOOST_PRESETS,
  VoiceBoostMode,
} from './voiceDsp';
import {
  DetectedMistake,
  MistakeKind,
  RealRecitationScore,
  ZipformerWordVerdict,
  analyzeSpokenVsExpected,
  computeRealRecitationScore,
  detectCompletedMistakesFromVerdicts,
  diagnosePerWordPhonemeMakhraj,
  evaluateTajweedFindingsForMistakes,
  getReciterAyahAudioUrl,
  judgeAttemptTilawi,
  normalizeZipformerTranscript,
  overlapMerge,
} from './mistakeDetector';
import {
  RawEngineTajweedItem,
  RecitationSessionRecord,
  aggregateWeakAreas,
  appendSessionHistory,
  buildSessionRecord,
  clearRecitationHistoryStorage,
  formatTajweedFindingAz,
  getArticulationTipAz,
  loadRecitationHistory,
  saveRecitationHistory,
} from './sessionAnalytics';

export interface CueWordInfo {
  surah: number;
  ayah: number;
  wordIndex: number;
  plain: string;
  mushaf: string;
}

export interface TartilMistakeRecord {
  s: number;
  a: number;
  w: number;
  kind: MistakeKind;
  n: number;
  t: number;
  note?: string;
}

export interface TartilGlobalApi {
  getPage: () => number;
  goToPage: (page: number) => void;
  isMicOn: () => boolean;
  toggleMic: () => void;
  preloadModel: () => Promise<unknown>;
  confirmWords: (surah: number, ayah: number, wordIndices: number[]) => void;
  getCueWord: () => CueWordInfo | null;
  getAyahPlainWords: (surah: number, ayah: number) => string[];
  revealNextWords: (count?: number) => number;
  flagMistake?: (issue: {
    surah: number;
    ayah: number;
    word: number;
    kind: MistakeKind;
    note?: string;
    expectedWord?: string;
    heardWord?: string;
    titleAz?: string;
  }) => void;
  enrichMistakeNote?: (surah: number, ayah: number, word: number, note: string) => boolean;
  retrySpot?: (surah: number, ayah: number, word?: number) => void;
  getMistakes?: () => TartilMistakeRecord[];
  clearMistakes?: () => void;
  getVerdicts?: () => ZipformerWordVerdict[];
  getTranscript?: () => string;
  markWarnWord?: (surah: number, ayah: number, word: number) => void;
  hasClip?: (surah: number, ayah: number) => boolean;
  playMine?: (surah: number, ayah: number) => void;
  getState: () => {
    page: number;
    mic: boolean;
    mistakes: number;
    revealedCount: number;
    sense?: string;
  };
}

export interface WorkletVoiceConfig {
  maxGain: number;
  gate: number;
  targetRms: number;
}

export type ReciterId = 'Husary_128kbps' | 'Alafasy_128kbps' | 'Minshawy_Murattal_128kbps';

declare global {
  interface Window {
    __tartilApi?: TartilGlobalApi;
    __tartilVoiceConfig?: WorkletVoiceConfig;
    __setTartilVoiceConfig?: (cfg: WorkletVoiceConfig) => void;
    __onTartilRms?: (rms: number) => void;
    __tj?: {
      found?: RawEngineTajweedItem[];
      stats?: number[];
    };
    __tartilMistakeTester?: {
      simulateSpokenArabic: (spokenTranscript: string) => {
        matchedIndices: number[];
        mistakes: DetectedMistake[];
      };
      judgeAyahAttempt: (surah: number, ayah: number, heardWords: string[]) => ReturnType<typeof judgeAttemptTilawi>;
      getRecentAlerts: () => MistakeAlertItem[];
      getRealScore: () => RealRecitationScore;
      getSessionHistory: () => RecitationSessionRecord[];
      clearAlerts: () => void;
      playReciterAyah: (surah: number, ayah: number) => Promise<void>;
    };
    SpeechRecognition?: any;
    webkitSpeechRecognition?: any;
  }
}

export interface MistakeAlertItem {
  surah: number;
  ayah: number;
  word: number;
  kind: MistakeKind;
  surahName: string;
  titleAz: string;
  explanationAz: string;
  expectedWord: string;
  heardWord?: string;
  timestamp: number;
}

const STORAGE_KEY = 'tartil.voiceBoost';
const SOUND_ALERT_KEY = 'tartil.soundAlert';
const STOP_ON_SLIP_KEY = 'tartil.stopOnSlip';
const RECITER_KEY = 'tartil.reciter';

export function getSavedVoiceBoost(): VoiceBoostMode {
  try {
    const saved = localStorage.getItem(STORAGE_KEY) as VoiceBoostMode | null;
    if (saved && saved in VOICE_BOOST_PRESETS) {
      return saved;
    }
  } catch {
    // ignore storage errors
  }
  return 'çox_güclü';
}

export function setSavedVoiceBoost(mode: VoiceBoostMode): void {
  const preset = VOICE_BOOST_PRESETS[mode] ?? VOICE_BOOST_PRESETS.çox_güclü;
  const cfg: WorkletVoiceConfig = {
    maxGain: preset.maxAdaptiveGain,
    gate: preset.noiseGateRms,
    targetRms: preset.targetRms,
  };
  window.__tartilVoiceConfig = cfg;
  window.__setTartilVoiceConfig?.(cfg);
  try {
    localStorage.setItem(STORAGE_KEY, mode);
    localStorage.setItem('tartil.gain', String(Math.round(preset.maxAdaptiveGain * 0.45)));
  } catch {
    // ignore storage errors
  }
}

export function getSavedReciter(): ReciterId {
  try {
    const r = localStorage.getItem(RECITER_KEY) as ReciterId | null;
    if (r === 'Husary_128kbps' || r === 'Alafasy_128kbps' || r === 'Minshawy_Murattal_128kbps') {
      return r;
    }
  } catch {
    // ignore
  }
  return 'Husary_128kbps';
}

/**
 * Plays a short, clear dual-tone warning chime using Web Audio API when a recitation mistake is caught.
 */
function playMistakeAlertTone(kind: MistakeKind): void {
  try {
    if (localStorage.getItem(SOUND_ALERT_KEY) === '0') return;
    const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
    if (!AudioCtx) return;
    const ctx = new AudioCtx();
    const now = ctx.currentTime;

    const osc1 = ctx.createOscillator();
    const osc2 = ctx.createOscillator();
    const gain = ctx.createGain();

    const baseFreq =
      kind === 'possible_vowel'
        ? 520
        : kind === 'possible_omission'
          ? 440
          : 390;

    osc1.type = 'triangle';
    osc2.type = 'sine';
    osc1.frequency.setValueAtTime(baseFreq, now);
    osc1.frequency.exponentialRampToValueAtTime(baseFreq * 0.78, now + 0.18);
    osc2.frequency.setValueAtTime(baseFreq * 1.5, now);
    osc2.frequency.exponentialRampToValueAtTime(baseFreq * 1.15, now + 0.18);

    gain.gain.setValueAtTime(0.001, now);
    gain.gain.linearRampToValueAtTime(0.16, now + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.22);

    osc1.connect(gain);
    osc2.connect(gain);
    gain.connect(ctx.destination);

    osc1.start(now);
    osc2.start(now);
    osc1.stop(now + 0.24);
    osc2.stop(now + 0.24);
    window.setTimeout(() => {
      ctx.close().catch(() => {});
    }, 300);
  } catch {
    // ignore audio context restrictions
  }
}

function getKindBadgeInfo(kind: MistakeKind): { badgeText: string; badgeClass: string } {
  switch (kind) {
    case 'possible_vowel':
      return { badgeText: 'Hərəkə xətası', badgeClass: 'kind-vowel' };
    case 'possible_omission':
      return { badgeText: 'Söz buraxıldı', badgeClass: 'kind-omit' };
    case 'possible_repetition':
      return { badgeText: 'Təkrar xətası', badgeClass: 'kind-repeat' };
    case 'possible_skipped_ayah':
      return { badgeText: 'Ayə buraxıldı', badgeClass: 'kind-omit' };
    default:
      return { badgeText: 'Məxrəc / Söz səhvi', badgeClass: 'kind-sub' };
  }
}

/**
 * Attaches:
 * 1. Voice Boost controls ("Güclü / Çox güclü / Ultra")
 * 2. Minhajy & Tilawa.ai Live Mistake Alert Banner (`.t-alert-banner`) with:
 *    - Expected vs Heard ("Said first, then retried") Arabic words
 *    - "🔊 Qarini dinlə (Şeyx Husari / Əfasi)" reference recitation audio player
 *    - "İndi düzəlt (Təkrar oxu)" single-click spot retry
 * 3. @tilawi/quran-asr Live Accuracy Score badge (`Dəqiqlik: %`) in the bottom bar
 * 4. Minhajy "Xətada saxla" (Stop on slip) strict mode option in Settings
 */
export function attachHybridVoiceBooster(rootEl: HTMLElement): () => void {
  let currentBoost: VoiceBoostMode = getSavedVoiceBoost();
  setSavedVoiceBoost(currentBoost);

  // Immediately preload local /corpus.json into window.__tartilCorpus so getCueWord & getAyahPlainWords work right away
  if (!(window as any).__tartilCorpus) {
    fetch('/corpus.json')
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (data && !(window as any).__tartilCorpus) {
          (window as any).__tartilCorpus = data;
        }
      })
      .catch(() => {});
  }

  let stopOnSlip = false;
  try {
    stopOnSlip = localStorage.getItem(STOP_ON_SLIP_KEY) === '1';
  } catch {
    // ignore
  }

  let currentReciter: ReciterId = getSavedReciter();
  let activeAudio: HTMLAudioElement | null = null;

  const playReciterAyah = async (surah: number, ayah: number, btnEl?: HTMLButtonElement | null) => {
    try {
      if (activeAudio) {
        activeAudio.pause();
        activeAudio = null;
      }
      const url = getReciterAyahAudioUrl(surah, ayah, currentReciter);
      const audio = new Audio(url);
      activeAudio = audio;
      if (btnEl) {
        btnEl.dataset.playing = 'true';
        btnEl.textContent = '🔊 Oxunur...';
      }
      audio.onended = () => {
        if (btnEl) {
          btnEl.dataset.playing = 'false';
          btnEl.textContent = '🔊 Qarini dinlə';
        }
      };
      audio.onerror = () => {
        if (btnEl) {
          btnEl.dataset.playing = 'false';
          btnEl.textContent = '🔊 Qarini dinlə';
        }
      };
      await audio.play();
    } catch {
      if (btnEl) {
        btnEl.dataset.playing = 'false';
        btnEl.textContent = '🔊 Qarini dinlə';
      }
    }
  };

  // Ensure default sensitivity is set to 'cetin' (Ultra 100x checking: Tajwid + Letters + GOP + Slip)
  try {
    if (!localStorage.getItem('tartil.sense')) {
      localStorage.setItem('tartil.sense', 'cetin');
    }
  } catch {
    // ignore
  }

  // 1. Inject "Səs tutma gücü", "Qari səsi (Minhajy/Tilawa)", and "Xətada saxla" rows into Settings (.t-sheet-box)
  const settingsBox = rootEl.querySelector('.t-sheet:not(.t-nav) .t-sheet-box');
  const senseHint =
    settingsBox?.querySelector('.t-sense-hint') ??
    settingsBox?.querySelector('.t-seg[data-set="sense"]')?.closest('.t-set');

  let boostRow: HTMLDivElement | null = null;
  let reciterRow: HTMLDivElement | null = null;
  let slipHoldRow: HTMLDivElement | null = null;

  if (settingsBox && senseHint && !settingsBox.querySelector('[data-set="voice-boost"]')) {
    boostRow = document.createElement('div');
    boostRow.className = 't-set';
    boostRow.innerHTML = `
      <span>Səs tutma gücü</span>
      <div class="t-seg" data-set="voice-boost">
        <button type="button" data-v="güclü" aria-pressed="${currentBoost === 'güclü'}">Güclü (2x)</button>
        <button type="button" data-v="çox_güclü" aria-pressed="${currentBoost === 'çox_güclü'}">Çox güclü (4x)</button>
        <button type="button" data-v="ultra" aria-pressed="${currentBoost === 'ultra'}">Ultra (8x)</button>
      </div>
    `;
    senseHint.insertAdjacentElement('afterend', boostRow);

    reciterRow = document.createElement('div');
    reciterRow.className = 't-set';
    reciterRow.innerHTML = `
      <span>Nümunə qari (Qarini dinlə)</span>
      <div class="t-seg" data-set="reciter-pick">
        <button type="button" data-r="Husary_128kbps" aria-pressed="${currentReciter === 'Husary_128kbps'}">Husari</button>
        <button type="button" data-r="Minshawy_Murattal_128kbps" aria-pressed="${currentReciter === 'Minshawy_Murattal_128kbps'}">Minşəvi</button>
        <button type="button" data-r="Alafasy_128kbps" aria-pressed="${currentReciter === 'Alafasy_128kbps'}">Əfasi</button>
      </div>
    `;
    boostRow.insertAdjacentElement('afterend', reciterRow);

    slipHoldRow = document.createElement('div');
    slipHoldRow.className = 't-set';
    slipHoldRow.innerHTML = `
      <span>Xətada avtomatik saxla <small>(Minhajy rejimi)</small></span>
      <div class="t-seg" data-set="stop-on-slip">
        <button type="button" data-s="0" aria-pressed="${!stopOnSlip}">Davam et</button>
        <button type="button" data-s="1" aria-pressed="${stopOnSlip}">Xətada saxla</button>
      </div>
    `;
    reciterRow.insertAdjacentElement('afterend', slipHoldRow);
  }

  // 2. Inject quick Voice Boost toggle, Hifz (Əzbər) toggle, & Real Acoustic Score Pill in the bottom bar (.t-bar)
  const bottomBar = rootEl.querySelector('.t-bar');
  let barBoostBtn: HTMLButtonElement | null = null;
  let barHifzBtn: HTMLButtonElement | null = null;
  let barScorePill: HTMLButtonElement | null = null;

  const isHifzActive = () => rootEl.classList.contains('hifz');
  const syncHifzPillUi = () => {
    if (!barHifzBtn) return;
    const on = isHifzActive();
    barHifzBtn.classList.toggle('active', on);
    barHifzBtn.setAttribute('aria-pressed', on ? 'true' : 'false');
    barHifzBtn.textContent = on ? 'Əzbər ✓' : 'Əzbər';
  };

  if (bottomBar && !bottomBar.querySelector('.t-boost-pill')) {
    const pillWrap = document.createElement('div');
    pillWrap.className = 't-bar-pills';

    barBoostBtn = document.createElement('button');
    barBoostBtn.type = 'button';
    barBoostBtn.dir = 'ltr';
    barBoostBtn.className = `t-boost-pill ${currentBoost === 'ultra' ? 'ultra' : ''}`;
    barBoostBtn.setAttribute('aria-label', 'Səs tutma gücü');
    barBoostBtn.title = 'Mikrofon səs tutma gücünü dəyişdir (Güclü / Çox güclü / Ultra)';

    barHifzBtn = document.createElement('button');
    barHifzBtn.type = 'button';
    barHifzBtn.dir = 'ltr';
    barHifzBtn.className = 't-hifz-pill';
    barHifzBtn.title = 'Hifz (Əzbər) rejimini aç/bağla — sözləri gizlədərək oxu';
    syncHifzPillUi();

    barScorePill = document.createElement('button');
    barScorePill.type = 'button';
    barScorePill.dir = 'ltr';
    barScorePill.id = 'tartil-accuracy-badge';
    barScorePill.className = 't-score-pill idle';
    barScorePill.title = 'Real akustik nəticəni, təcvidi və tarixçəni aç';
    barScorePill.textContent = 'Nəticə: —';

    pillWrap.appendChild(barBoostBtn);
    pillWrap.appendChild(barHifzBtn);
    pillWrap.appendChild(barScorePill);
    bottomBar.appendChild(pillWrap);
  }

  // Session verdict cache preserves real acoustic metrics across ayah/page transitions
  const sessionVerdictCache = new Map<string, ZipformerWordVerdict>();
  const autoFlaggedKeys = new Set<string>();
  let lastLiveAcousticTranscript = '';
  let stitchedWindowWords: string[] = [];
  let prevMicState = false;
  let telemetryTab: 'current' | 'history' = 'current';
  let isGuidedDrillActive = false;
  let sessionHistory: RecitationSessionRecord[] = loadRecitationHistory();

  // Create Real Acoustic Telemetry Sheet (.t-telemetry-sheet)
  let telemetrySheet = rootEl.querySelector('.t-telemetry-sheet') as HTMLDivElement | null;
  if (!telemetrySheet) {
    telemetrySheet = document.createElement('div');
    telemetrySheet.className = 't-sheet t-telemetry-sheet';
    telemetrySheet.hidden = true;
    telemetrySheet.setAttribute('role', 'dialog');
    telemetrySheet.setAttribute('aria-label', 'Real akustik nəticələr');
    rootEl.appendChild(telemetrySheet);
  }

  const getWordPlainText = (surah: number, ayah: number, wordIdx: number): string => {
    const words = window.__tartilApi?.getAyahPlainWords?.(surah, ayah);
    return words && words[wordIdx] ? words[wordIdx] : `Söz ${wordIdx + 1}`;
  };

  const syncVerdictsFromEngine = (
    isSessionEnding = false,
    overrideVerdicts?: ZipformerWordVerdict[],
    overrideTranscript?: string
  ) => {
    const api = window.__tartilApi;
    if (!api) return;
    const liveVerdicts = overrideVerdicts ?? api.getVerdicts?.() ?? [];
    for (const v of liveVerdicts) {
      if (!v || !v.surah || !v.ayah || v.state === 'pending') continue;
      const key = `${v.surah}:${v.ayah}:${v.word}`;
      sessionVerdictCache.set(key, v);
      if (v.state === 'unsure') {
        api.markWarnWord?.(v.surah, v.ayah, v.word);
      }
    }
    const tr = overrideTranscript ?? api.getTranscript?.() ?? '';
    if (tr && tr.trim()) {
      lastLiveAcousticTranscript = tr.trim();
    }

    // Evaluate completed ayahs / multi-word skipped spans on settled session verdicts
    if (api.isMicOn() || isSessionEnding) {
      const completedMistakes = detectCompletedMistakesFromVerdicts(
        Array.from(sessionVerdictCache.values()),
        getWordPlainText,
        { isSessionEnding, transcript: lastLiveAcousticTranscript }
      );
      const existing = api.getMistakes?.() ?? [];
      for (const m of completedMistakes) {
        const mKey = `${m.surah}:${m.ayah}:${m.word}`;
        if (autoFlaggedKeys.has(mKey)) continue;
        const already = existing.find((e) => e.s === m.surah && e.a === m.ayah && e.w === m.word);
        if (already) {
          if (!already.note && m.explanationAz) {
            api.enrichMistakeNote?.(m.surah, m.ayah, m.word, m.explanationAz);
          }
          autoFlaggedKeys.add(mKey);
          continue;
        }
        autoFlaggedKeys.add(mKey);
        api.flagMistake?.({
          surah: m.surah,
          ayah: m.ayah,
          word: m.word,
          kind: m.kind,
          note: m.explanationAz,
          expectedWord: m.expectedWord,
          heardWord: m.heardWord,
          titleAz: m.titleAz,
        });
      }
    }
  };

  const computeCurrentRealScore = (): RealRecitationScore => {
    const api = window.__tartilApi;
    const st = api?.getState?.();
    const activeMistakes = api?.getMistakes?.() ?? [];
    const mistakesCount = st?.mistakes ?? activeMistakes.length;
    const flaggedMap = new Map<string, MistakeKind>();
    for (const m of activeMistakes) {
      if (m && m.s && m.a && typeof m.w === 'number' && m.w >= 0) {
        flaggedMap.set(`${m.s}:${m.a}:${m.w}`, m.kind);
      }
    }
    return computeRealRecitationScore(
      Array.from(sessionVerdictCache.values()),
      mistakesCount,
      getWordPlainText,
      flaggedMap
    );
  };

  const persistCompletedSession = () => {
    const score = computeCurrentRealScore();
    const api = window.__tartilApi;
    const activeMistakes = api?.getMistakes?.() ?? [];
    const nextRec = buildSessionRecord({
      page: api?.getPage?.() ?? 1,
      score,
      mistakes: activeMistakes,
    });
    if (!nextRec) return;
    const updated = appendSessionHistory(sessionHistory, nextRec);
    if (updated !== sessionHistory) {
      sessionHistory = updated;
      saveRecitationHistory(sessionHistory);
    }
  };

  const renderTelemetrySheet = () => {
    if (!telemetrySheet || telemetrySheet.hidden) return;
    const score = computeCurrentRealScore();
    const asrSpeech = stitchedWindowWords.slice(-18).join(' ').trim();
    const acousticText = lastLiveAcousticTranscript || asrSpeech;
    const api = window.__tartilApi;
    const activeMistakes = api?.getMistakes?.() ?? [];

    const rowsHtml =
      score.breakdowns.length === 0
        ? `<div class="t-tel-empty">Hələ heç bir söz akustik model tərəfindən qiymətləndirilməyib. Mikrofonu açıb qiraətə başladıqda burada hər sözün real məsafəsi, GOP balı və dəqiq faizi görünəcək.</div>`
        : score.breakdowns
            .slice()
            .reverse()
            .map((b) => {
              const isSevereTag =
                b.reasonTag === 'Məxrəc' ||
                b.reasonTag === 'Buraxılıb' ||
                b.reasonTag === 'Xəta' ||
                b.reasonTag === 'GOP' ||
                b.reasonTag === 'Neyron';
              const badgeCls =
                b.reasonTag === 'Təkrar'
                  ? 'repeat'
                  : b.state === 'wrong' || b.state === 'skipped' || isSevereTag
                    ? 'bad'
                    : b.reasonTag === 'Hərəkə' || b.state === 'unsure' || b.wordScorePercent < 82
                      ? 'warn'
                      : 'ok';
              const badgeLbl = b.reasonTag || (b.wordScorePercent < 82 ? 'Zəif tələffüz' : 'Təmiz');
              const gopStr = b.gop !== null ? `GOP ${b.gop.toFixed(1)}` : 'GOP —';
              const repStr = b.repGain !== null && b.repGain >= 3.5 ? ` · Təkrar +${b.repGain.toFixed(1)}` : '';
              const showHeard =
                badgeCls !== 'ok' &&
                b.heard &&
                b.expected &&
                normalizeZipformerTranscript(b.heard) !== normalizeZipformerTranscript(b.expected);
              const heardStr = showHeard ? ` · eşidilən: <span dir="rtl">${b.heard}</span>` : '';
              const distStr = `məsafə ${b.distance.toFixed(2)}`;
              const hasUserClip = Boolean(api?.hasClip?.(b.surah, b.ayah));
              const mineBtn =
                badgeCls !== 'ok' && hasUserClip
                  ? `<button type="button" class="t-tel-mine" data-s="${b.surah}" data-a="${b.ayah}" title="Öz səsimi dinlə">🎙️</button>`
                  : '';
              const listenBtn =
                badgeCls !== 'ok'
                  ? `<button type="button" class="t-tel-play" data-s="${b.surah}" data-a="${b.ayah}" title="Qarini dinlə">🔊</button>`
                  : '';
              return `
                <div class="t-tel-row">
                  <span class="t-tel-word" dir="rtl">${b.wordText}</span>
                  <span class="t-tel-loc">${b.surah}:${b.ayah} · #${b.word + 1}</span>
                  <span class="t-tel-meta">${distStr} · ${gopStr}${repStr}${heardStr}</span>
                  <span class="t-tel-badge ${badgeCls}">${badgeLbl} · ${b.wordScorePercent}%</span>
                  ${mineBtn}${listenBtn}
                </div>
              `;
            })
            .join('');

    const rawTjFound = window.__tj?.found ?? [];
    const tjItems = rawTjFound.slice(-6).reverse().map(formatTajweedFindingAz);
    const tajweedSectionHtml =
      tjItems.length > 0
        ? `
          <div class="t-tel-tj">
            <span class="t-tel-stream-lbl">Təcvid Qaydaları Yoxlaması (${rawTjFound.length}):</span>
            <div class="t-tel-tj-list">
              ${tjItems
                .map(
                  (t) => `
                  <button type="button" class="t-tel-tj-chip" data-s="${t.surah}" data-a="${t.ayah}" data-w="${t.word}" title="${t.detailAz}">
                    <span dir="rtl" class="t-tel-tj-w">${t.wordText}</span>
                    <span>${t.ruleLabelAz} · ${t.surah}:${t.ayah}</span>
                  </button>`
                )
                .join('')}
            </div>
          </div>`
        : '';

    const { weakAreas, recurringSpots } = aggregateWeakAreas(sessionHistory);
    const historyBodyHtml =
      sessionHistory.length === 0
        ? `<div class="t-tel-empty">Hələ qeyd edilmiş təlim sessiyası yoxdur. Qiraət edib mikrofonu söndürdüyünüzdə real nəticələr və çətinlik çəkdiyiniz məxrəclər burada toplanacaq.</div>`
        : `
          ${
            weakAreas.length > 0
              ? `
            <div class="t-tel-weak-box">
              <div class="t-tel-subhead">Ən Çox Təkrarlanan Zəif Mövzular</div>
              <div class="t-tel-weak-list">
                ${weakAreas
                  .map(
                    (wa) => `
                  <div class="t-tel-weak-item">
                    <div class="t-tel-weak-head">
                      <b><bdi>${wa.tag}</bdi></b>
                      <span>${wa.count} dəfə</span>
                    </div>
                    <p class="t-tel-weak-tip">${wa.tipAz}</p>
                  </div>`
                  )
                  .join('')}
              </div>
            </div>`
              : ''
          }
          ${
            recurringSpots.length > 0
              ? `
            <div class="t-tel-spots-box">
              <div class="t-tel-subhead">Məşq tələb edən sözlər (kliklə → məşq et)</div>
              <div class="t-tel-tj-list">
                ${recurringSpots
                  .map(
                    (sp) => `
                  <button type="button" class="t-tel-spot-chip" data-s="${sp.surah}" data-a="${sp.ayah}" data-w="${sp.word}">
                    <bdi dir="rtl" class="t-tel-tj-w">${sp.wordText}</bdi>
                    <span><bdi>${sp.surah}:${sp.ayah}</bdi> · <bdi>${sp.tag}</bdi> (${sp.count}x)</span>
                  </button>`
                  )
                  .join('')}
              </div>
            </div>`
              : ''
          }
          <div class="t-tel-subhead-row">
            <span class="t-tel-subhead">Son Qiraət Sessiyaları</span>
            <button type="button" class="t-tel-clear-hist">Tarixçəni təmizlə</button>
          </div>
          <div class="t-tel-list">
            ${sessionHistory
              .map((h) => {
                const dtStr = new Date(h.timestamp).toLocaleTimeString('az-AZ', {
                  hour: '2-digit',
                  minute: '2-digit',
                });
                const cls = h.scorePercent >= 88 ? 'ok' : h.scorePercent >= 70 ? 'warn' : 'bad';
                return `
                  <div class="t-tel-row">
                    <span class="t-tel-loc">Səh. ${h.page} (${h.surah}:${h.ayahFrom}${h.ayahTo > h.ayahFrom ? `–${h.ayahTo}` : ''})</span>
                    <span class="t-tel-meta">${dtStr} · ${h.evaluatedWords} söz · ${h.errorWords} xəta</span>
                    <span class="t-tel-badge ${cls}">${h.scorePercent}%</span>
                    <button type="button" class="t-tel-go-page" data-p="${h.page}">Keç</button>
                  </div>`;
              })
              .join('')}
          </div>`;

    const practiceBtnHtml =
      activeMistakes.length > 0
        ? `<button type="button" class="t-tel-practice">Xətalı yerləri məşq et (${activeMistakes.length})</button>`
        : '';

    const currentBodyHtml = `
      <div class="t-tel-summary">
        <div class="t-tel-kpi">
          <span>Real Dəqiqlik</span>
          <b>${score.hasData && score.scorePercent !== null ? `${score.scorePercent}%` : '—'}</b>
        </div>
        <div class="t-tel-kpi">
          <span>Söz sayı</span>
          <b>${score.evaluatedWords}</b>
        </div>
        <div class="t-tel-kpi">
          <span>Təmiz / Şübhəli / Xəta</span>
          <b>${score.cleanWords} / ${score.unsureWords} / ${score.errorWords}</b>
        </div>
        <div class="t-tel-kpi">
          <span>Orta Məsafə / GOP</span>
          <b>${score.hasData ? `${score.avgDistance.toFixed(2)} / ${score.avgGop !== null ? score.avgGop.toFixed(1) : '—'}` : '—'}</b>
        </div>
      </div>
      <div class="t-tel-stream">
        <span class="t-tel-stream-lbl">Modelin eşitdiyi canlı axın:</span>
        <div class="t-tel-stream-val" dir="rtl">${acousticText || 'Səs gözlənilir…'}</div>
      </div>
      ${tajweedSectionHtml}
      <div class="t-tel-list">${rowsHtml}</div>
    `;

    telemetrySheet.innerHTML = `
      <div class="t-sheet-box t-tel-box">
        <div class="t-sheet-h">
          <b>Real Akustik Nəticələr</b>
          <div class="t-tel-head-actions">
            ${practiceBtnHtml}
            <button type="button" class="t-sheet-x">Bağla</button>
          </div>
        </div>
        <div class="t-tel-tabs" role="tablist">
          <button type="button" class="t-tel-tab ${telemetryTab === 'current' ? 'active' : ''}" data-tab="current">Bu Qiraət</button>
          <button type="button" class="t-tel-tab ${telemetryTab === 'history' ? 'active' : ''}" data-tab="history">Tarixçə və Zəif Məxrəclər (${sessionHistory.length})</button>
        </div>
        ${telemetryTab === 'current' ? currentBodyHtml : historyBodyHtml}
      </div>
    `;

    telemetrySheet.querySelector('.t-sheet-x')?.addEventListener('click', () => {
      if (telemetrySheet) telemetrySheet.hidden = true;
    });
    telemetrySheet.querySelectorAll('.t-tel-tab').forEach((el) => {
      el.addEventListener('click', () => {
        const nextTab = (el as HTMLButtonElement).dataset.tab;
        if (nextTab === 'current' || nextTab === 'history') {
          telemetryTab = nextTab;
          renderTelemetrySheet();
        }
      });
    });
    telemetrySheet.querySelector('.t-tel-practice')?.addEventListener('click', () => {
      const first = window.__tartilApi?.getMistakes?.()?.[0];
      if (telemetrySheet) telemetrySheet.hidden = true;
      if (first) {
        isGuidedDrillActive = true;
        window.__tartilApi?.retrySpot?.(first.s, first.a, first.w);
      }
    });
    telemetrySheet.querySelectorAll('.t-tel-play').forEach((el) => {
      el.addEventListener('click', () => {
        const btn = el as HTMLButtonElement;
        const s = Number(btn.dataset.s || 0);
        const a = Number(btn.dataset.a || 0);
        if (s && a) playReciterAyah(s, a, btn);
      });
    });
    telemetrySheet.querySelectorAll('.t-tel-mine').forEach((el) => {
      el.addEventListener('click', () => {
        const btn = el as HTMLButtonElement;
        const s = Number(btn.dataset.s || 0);
        const a = Number(btn.dataset.a || 0);
        if (s && a) window.__tartilApi?.playMine?.(s, a);
      });
    });
    telemetrySheet.querySelectorAll('.t-tel-tj-chip, .t-tel-spot-chip').forEach((el) => {
      el.addEventListener('click', () => {
        const btn = el as HTMLButtonElement;
        const s = Number(btn.dataset.s || 0);
        const a = Number(btn.dataset.a || 0);
        const w = Number(btn.dataset.w ?? -1);
        if (telemetrySheet) telemetrySheet.hidden = true;
        if (s && a) window.__tartilApi?.retrySpot?.(s, a, w);
      });
    });
    telemetrySheet.querySelectorAll('.t-tel-go-page').forEach((el) => {
      el.addEventListener('click', () => {
        const p = Number((el as HTMLButtonElement).dataset.p || 0);
        if (telemetrySheet) telemetrySheet.hidden = true;
        if (p > 0) window.__tartilApi?.goToPage?.(p);
      });
    });
    telemetrySheet.querySelector('.t-tel-clear-hist')?.addEventListener('click', () => {
      sessionHistory = [];
      clearRecitationHistoryStorage();
      renderTelemetrySheet();
    });
  };

  telemetrySheet.addEventListener('click', (ev) => {
    if (ev.target === telemetrySheet && telemetrySheet) {
      telemetrySheet.hidden = true;
    }
  });

  barScorePill?.addEventListener('click', () => {
    if (!telemetrySheet) return;
    syncVerdictsFromEngine();
    telemetrySheet.hidden = !telemetrySheet.hidden;
    if (!telemetrySheet.hidden) {
      renderTelemetrySheet();
    }
  });

  const updateScorePill = () => {
    if (!barScorePill) return;
    syncVerdictsFromEngine();
    const score = computeCurrentRealScore();
    if (!score.hasData || score.scorePercent === null) {
      barScorePill.textContent = 'Nəticə: —';
      barScorePill.classList.add('idle');
      barScorePill.classList.remove('warn', 'bad');
    } else {
      const pct = score.scorePercent;
      barScorePill.textContent = `${pct}% · ${score.evaluatedWords} söz`;
      barScorePill.classList.remove('idle');
      barScorePill.classList.toggle('warn', pct < 88 && pct >= 70);
      barScorePill.classList.toggle('bad', pct < 70);
    }
    if (telemetrySheet && !telemetrySheet.hidden) {
      renderTelemetrySheet();
    }
  };

  // 3. Create Live Mistake Radar Banner (.t-alert-banner) anchored above the bottom bar
  let alertBanner = rootEl.querySelector('.t-alert-banner') as HTMLDivElement | null;
  if (!alertBanner) {
    alertBanner = document.createElement('div');
    alertBanner.className = 't-alert-banner';
    alertBanner.hidden = true;
    alertBanner.setAttribute('role', 'status');
    alertBanner.setAttribute('aria-live', 'assertive');
    rootEl.appendChild(alertBanner);
  }

  const recentAlerts: MistakeAlertItem[] = [];
  let bannerHideTimer: number | null = null;

  const renderAlertBanner = (item: MistakeAlertItem) => {
    if (!alertBanner) return;
    if (bannerHideTimer) {
      window.clearTimeout(bannerHideTimer);
      bannerHideTimer = null;
    }

    const { badgeText, badgeClass } = getKindBadgeInfo(item.kind);
    const wordPosText = item.word >= 0 ? ` · Söz ${item.word + 1}` : '';
    const heardPart = item.heardWord
      ? `<span class="t-alert-heard" dir="ltr">Eşidilən: <b dir="rtl">${item.heardWord}</b></span>`
      : '';
    const tipText = getArticulationTipAz({
      kind: item.kind,
      expectedWord: item.expectedWord,
      heardWord: item.heardWord,
      note: item.explanationAz,
    });
    const hasMyClip = Boolean(window.__tartilApi?.hasClip?.(item.surah, item.ayah));
    const mineBtnHtml = hasMyClip
      ? `<button type="button" class="t-alert-mine" title="Bu ayədə öz qiraət səsinizi dinləyin">🎙️ Öz səsim</button>`
      : '';

    alertBanner.innerHTML = `
      <div class="t-alert-main">
        <div class="t-alert-top">
          <span class="t-alert-badge ${badgeClass}">${badgeText}</span>
          <span class="t-alert-loc">${item.surahName} ${item.ayah}${wordPosText}</span>
          <button type="button" class="t-alert-close" aria-label="Bağla">✕</button>
        </div>
        <div class="t-alert-body">
          <div class="t-alert-words">
            <span class="t-alert-expected" dir="ltr">Düzgün: <b dir="rtl">${item.expectedWord || '—'}</b></span>
            ${heardPart}
          </div>
          <p class="t-alert-desc" dir="ltr">${item.explanationAz}</p>
          <p class="t-alert-tip" dir="ltr"><b>💡 Məsləhət:</b> ${tipText}</p>
        </div>
        <div class="t-alert-actions">
          ${mineBtnHtml}
          <button type="button" class="t-alert-listen" title="Şeyx Husari / Minşəvi / Əfasi tilavətini dinlə">🔊 Qarini dinlə</button>
          <button type="button" class="t-alert-retry">İndi düzəlt (Təkrar oxu)</button>
        </div>
      </div>
    `;
    alertBanner.hidden = false;

    const closeBtn = alertBanner.querySelector('.t-alert-close');
    closeBtn?.addEventListener('click', () => {
      if (alertBanner) alertBanner.hidden = true;
    });

    const mineBtn = alertBanner.querySelector('.t-alert-mine') as HTMLButtonElement | null;
    mineBtn?.addEventListener('click', () => {
      window.__tartilApi?.playMine?.(item.surah, item.ayah);
    });

    const listenBtn = alertBanner.querySelector('.t-alert-listen') as HTMLButtonElement | null;
    listenBtn?.addEventListener('click', () => {
      playReciterAyah(item.surah, item.ayah, listenBtn);
    });

    const retryBtn = alertBanner.querySelector('.t-alert-retry');
    retryBtn?.addEventListener('click', () => {
      if (alertBanner) alertBanner.hidden = true;
      window.__tartilApi?.retrySpot?.(item.surah, item.ayah, item.word);
    });

    bannerHideTimer = window.setTimeout(() => {
      if (alertBanner) alertBanner.hidden = true;
    }, 14000);
  };

  // Listen to `tartil-mistake` events emitted by `ke(e)` in `tartil-engine.js`
  const onTartilMistakeEvent = (ev: Event) => {
    const detail = (ev as CustomEvent).detail || {};
    const surah = Number(detail.surah || 0);
    const ayah = Number(detail.ayah || 0);
    const word = typeof detail.word === 'number' ? detail.word : -1;
    const kind = (detail.kind || 'possible_substitution') as MistakeKind;
    if (!surah || !ayah) return;

    const plainWords = window.__tartilApi?.getAyahPlainWords?.(surah, ayah) ?? [];
    const expectedWord =
      detail.expectedWord || (word >= 0 && plainWords[word] ? plainWords[word] : `Ayə ${ayah}`);

    const cachedVerdict = word >= 0 ? sessionVerdictCache.get(`${surah}:${ayah}:${word}`) : undefined;
    const perWordMakhraj =
      cachedVerdict && kind === 'possible_substitution'
        ? diagnosePerWordPhonemeMakhraj(cachedVerdict.expected, cachedVerdict.heard, {
            distance: cachedVerdict.distance,
            gop: cachedVerdict.gop,
            state: cachedVerdict.state,
          })
        : null;

    const resolvedHeard =
      detail.heardWord ||
      perWordMakhraj?.heardSegment ||
      (cachedVerdict?.heard && cachedVerdict.heard !== cachedVerdict.expected
        ? cachedVerdict.heard
        : undefined);

    const defaultExplanation =
      perWordMakhraj
        ? `\u200EMəxrəc xətası (${expectedWord}): ${perWordMakhraj.confusion.descriptionAz}`
        : kind === 'possible_vowel'
          ? `\u200EHərəkə xətası (${expectedWord}): fəthə/kəsrə/zəmmə yanlış tələffüz edildi`
          : kind === 'possible_omission'
            ? `\u200ESöz buraxıldı: «${expectedWord}» oxunmadan ötürüldü`
            : kind === 'possible_repetition'
              ? `\u200ETəkrar xətası: «${expectedWord}» sözü təkrar oxundu`
              : `\u200ETələffüz / Məxrəc xətası (${expectedWord}): hərf və ya söz səhv oxundu`;

    const alertItem: MistakeAlertItem = {
      surah,
      ayah,
      word,
      kind,
      surahName: detail.surahName || `Surə ${surah}`,
      titleAz:
        detail.titleAz ||
        (perWordMakhraj
          ? `Hərf məxrəci xətası (${perWordMakhraj.confusion.expectedLetter} → ${perWordMakhraj.confusion.heardLetter})`
          : detail.kindLabel || 'Tələffüz xətası'),
      explanationAz: detail.note || defaultExplanation,
      expectedWord,
      heardWord: resolvedHeard,
      timestamp: Date.now(),
    };

    recentAlerts.unshift(alertItem);
    if (recentAlerts.length > 50) recentAlerts.pop();

    playMistakeAlertTone(kind);
    renderAlertBanner(alertItem);
    updateScorePill();

    // If Minhajy "Stop on slip" mode is active, automatically hold on the mistaken word for immediate retry
    if (stopOnSlip && word >= 0) {
      window.setTimeout(() => {
        window.__tartilApi?.retrySpot?.(surah, ayah, word);
      }, 150);
    }
  };

  window.addEventListener('tartil-mistake', onTartilMistakeEvent);

  const onTartilFinalVerdicts = (ev: Event) => {
    const detail = (ev as CustomEvent).detail || {};
    const finalVerdicts = Array.isArray(detail.verdicts) ? detail.verdicts : [];
    const finalTranscript = typeof detail.transcript === 'string' ? detail.transcript : '';
    syncVerdictsFromEngine(true, finalVerdicts, finalTranscript);
    persistCompletedSession();
    updateScorePill();
  };
  window.addEventListener('tartil-final-verdicts', onTartilFinalVerdicts);

  const onTartilTjFound = (ev: Event) => {
    const api = window.__tartilApi;
    if (!api) return;
    const detail = (ev as CustomEvent).detail || {};
    const items = Array.isArray(detail.items) ? detail.items : [];
    if (items.length === 0) return;

    const promoted = evaluateTajweedFindingsForMistakes(
      items,
      getWordPlainText,
      Array.from(sessionVerdictCache.values())
    );
    if (promoted.length === 0) return;

    const existing = api.getMistakes?.() ?? [];
    for (const m of promoted) {
      const mKey = `${m.surah}:${m.ayah}:${m.word}`;
      if (autoFlaggedKeys.has(mKey)) continue;
      const already = existing.find((e) => e.s === m.surah && e.a === m.ayah && e.w === m.word);
      if (already) {
        if (!already.note && m.explanationAz) {
          api.enrichMistakeNote?.(m.surah, m.ayah, m.word, m.explanationAz);
        }
        autoFlaggedKeys.add(mKey);
        continue;
      }
      autoFlaggedKeys.add(mKey);
      api.flagMistake?.({
        surah: m.surah,
        ayah: m.ayah,
        word: m.word,
        kind: m.kind,
        note: m.explanationAz,
        expectedWord: m.expectedWord,
        heardWord: m.heardWord,
        titleAz: m.titleAz,
      });
    }
    updateScorePill();
  };
  window.addEventListener('tartil-tj-found', onTartilTjFound);

  const onTartilMistakesCleared = () => {
    recentAlerts.length = 0;
    sessionVerdictCache.clear();
    autoFlaggedKeys.clear();
    lastLiveAcousticTranscript = '';
    isGuidedDrillActive = false;
    if (alertBanner) alertBanner.hidden = true;
    updateScorePill();
  };
  window.addEventListener('tartil-mistakes-cleared', onTartilMistakesCleared);

  const onTartilMistakeFixed = (ev: Event) => {
    const detail = (ev as CustomEvent).detail || {};
    const surah = Number(detail.surah || 0);
    const ayah = Number(detail.ayah || 0);
    const word = typeof detail.word === 'number' ? detail.word : -1;
    if (!surah || !ayah) return;

    if (word >= 0) {
      const key = `${surah}:${ayah}:${word}`;
      autoFlaggedKeys.delete(key);
      const existingVerdict = sessionVerdictCache.get(key);
      if (existingVerdict && existingVerdict.state !== 'ok') {
        sessionVerdictCache.set(key, {
          ...existingVerdict,
          state: 'ok',
          distance: 0,
          heardRatio: 1,
          vowelErrors: 0,
          repGain: undefined,
          slip: undefined,
        });
      }
    } else {
      const prefix = `${surah}:${ayah}:`;
      for (const k of Array.from(autoFlaggedKeys)) {
        if (k.startsWith(prefix)) autoFlaggedKeys.delete(k);
      }
    }

    let removedTop = false;
    for (let idx = recentAlerts.length - 1; idx >= 0; idx--) {
      const item = recentAlerts[idx];
      if (item.surah === surah && item.ayah === ayah && (word < 0 || item.word === word)) {
        if (idx === 0) removedTop = true;
        recentAlerts.splice(idx, 1);
      }
    }
    if (removedTop && alertBanner) {
      alertBanner.hidden = true;
    }
    updateScorePill();

    // Guided Mistake Drill Loop: auto-step to the next remaining mistake if drill mode is active
    if (isGuidedDrillActive) {
      const remaining = window.__tartilApi?.getMistakes?.() ?? [];
      if (remaining.length > 0) {
        const nextSpot = remaining[0];
        window.setTimeout(() => {
          window.__tartilApi?.retrySpot?.(nextSpot.s, nextSpot.a, nextSpot.w);
        }, 550);
      } else {
        isGuidedDrillActive = false;
      }
    }
  };
  window.addEventListener('tartil-mistake-fixed', onTartilMistakeFixed);

  const updateBoostUi = (mode: VoiceBoostMode) => {
    currentBoost = mode;
    setSavedVoiceBoost(mode);
    const preset = VOICE_BOOST_PRESETS[mode];
    if (boostRow) {
      boostRow.querySelectorAll('button').forEach((btn) => {
        btn.setAttribute('aria-pressed', btn.dataset.v === mode ? 'true' : 'false');
      });
    }
    if (barBoostBtn) {
      const shortBoost = mode === 'ultra' ? 'Səs: 8x' : mode === 'çox_güclü' ? 'Səs: 4x' : 'Səs: 2x';
      barBoostBtn.textContent = shortBoost;
      barBoostBtn.classList.toggle('ultra', mode === 'ultra' || mode === 'çox_güclü');
    }
  };

  updateBoostUi(currentBoost);

  const onBoostRowClick = (ev: Event) => {
    const target = (ev.target as HTMLElement | null)?.closest('button[data-v]') as HTMLButtonElement | null;
    if (!target) return;
    const val = target.dataset.v as VoiceBoostMode;
    if (val && val in VOICE_BOOST_PRESETS) {
      updateBoostUi(val);
    }
  };
  boostRow?.addEventListener('click', onBoostRowClick);

  const onReciterRowClick = (ev: Event) => {
    const target = (ev.target as HTMLElement | null)?.closest('button[data-r]') as HTMLButtonElement | null;
    if (!target) return;
    const val = target.dataset.r as ReciterId;
    if (val) {
      currentReciter = val;
      try {
        localStorage.setItem(RECITER_KEY, val);
      } catch {
        // ignore
      }
      reciterRow?.querySelectorAll('button').forEach((btn) => {
        btn.setAttribute('aria-pressed', btn.dataset.r === val ? 'true' : 'false');
      });
    }
  };
  reciterRow?.addEventListener('click', onReciterRowClick);

  const onSlipHoldRowClick = (ev: Event) => {
    const target = (ev.target as HTMLElement | null)?.closest('button[data-s]') as HTMLButtonElement | null;
    if (!target) return;
    stopOnSlip = target.dataset.s === '1';
    try {
      localStorage.setItem(STOP_ON_SLIP_KEY, stopOnSlip ? '1' : '0');
    } catch {
      // ignore
    }
    slipHoldRow?.querySelectorAll('button').forEach((btn) => {
      btn.setAttribute('aria-pressed', (btn.dataset.s === '1') === stopOnSlip ? 'true' : 'false');
    });
  };
  slipHoldRow?.addEventListener('click', onSlipHoldRowClick);

  const onBarBoostClick = () => {
    const order: VoiceBoostMode[] = ['güclü', 'çox_güclü', 'ultra'];
    const next = order[(order.indexOf(currentBoost) + 1) % order.length];
    updateBoostUi(next);
  };
  barBoostBtn?.addEventListener('click', onBarBoostClick);

  const settingsHifzSeg = settingsBox?.querySelector('.t-seg[data-set="hifz"]') ?? null;
  const onBarHifzClick = () => {
    const nextVal = isHifzActive() ? '0' : '1';
    const targetBtn = settingsBox?.querySelector(
      `.t-seg[data-set="hifz"] button[data-v="${nextVal}"]`
    ) as HTMLButtonElement | null;
    if (targetBtn) {
      targetBtn.click();
    } else {
      rootEl.classList.toggle('hifz', nextVal === '1');
      try {
        localStorage.setItem('tartil.hifz', nextVal);
      } catch {
        // ignore
      }
    }
    syncHifzPillUi();
  };
  barHifzBtn?.addEventListener('click', onBarHifzClick);

  const onSettingsHifzClick = () => {
    window.setTimeout(syncHifzPillUi, 0);
  };
  settingsHifzSeg?.addEventListener('click', onSettingsHifzClick);

  // 4. Never advance words on RMS energy.
  window.__onTartilRms = undefined;

  // Helper that processes a spoken Arabic transcript against the current cue word & ayah
  const processSpokenArabicTranscript = (spokenTranscript: string) => {
    const api = window.__tartilApi;
    if (!api) return { matchedIndices: [], mistakes: [] };
    const cue = api.getCueWord?.();
    if (!cue) return { matchedIndices: [], mistakes: [] };
    const expectedWords = api.getAyahPlainWords?.(cue.surah, cue.ayah) ?? [];
    if (!expectedWords.length) return { matchedIndices: [], mistakes: [] };

    const analysis = analyzeSpokenVsExpected({
      surah: cue.surah,
      ayah: cue.ayah,
      expectedWords,
      currentWordIndex: cue.wordIndex,
      spokenTranscript,
    });

    for (const m of analysis.mistakes) {
      api.flagMistake?.({
        surah: m.surah,
        ayah: m.ayah,
        word: m.word,
        kind: m.kind,
        note: m.explanationAz,
        expectedWord: m.expectedWord,
        heardWord: m.heardWord,
        titleAz: m.titleAz,
      });
    }

    if (analysis.matchedIndices.length > 0) {
      api.confirmWords(cue.surah, cue.ayah, analysis.matchedIndices);
    }

    updateScorePill();
    return analysis;
  };

  // Expose deterministic tester API for verification and diagnostics
  window.__tartilMistakeTester = {
    simulateSpokenArabic: (spokenTranscript: string) =>
      processSpokenArabicTranscript(spokenTranscript),
    judgeAyahAttempt: (surah: number, ayah: number, heardWords: string[]) => {
      const expected = window.__tartilApi?.getAyahPlainWords?.(surah, ayah) ?? [];
      return judgeAttemptTilawi(expected, heardWords);
    },
    getRecentAlerts: () => [...recentAlerts],
    getRealScore: () => {
      syncVerdictsFromEngine();
      return computeCurrentRealScore();
    },
    getSessionHistory: () => [...sessionHistory],
    clearAlerts: () => {
      recentAlerts.length = 0;
      sessionVerdictCache.clear();
      autoFlaggedKeys.clear();
      lastLiveAcousticTranscript = '';
      if (alertBanner) alertBanner.hidden = true;
      window.__tartilApi?.clearMistakes?.();
      updateScorePill();
    },
    playReciterAyah: (surah: number, ayah: number) => playReciterAyah(surah, ayah),
  };

  // 5. Parallel Browser Web Speech API (ar-SA) Assist — strictly confirms ONLY final exact-matching words, NEVER flags false mistakes on Qari Tajweed/Madd
  let recognition: any = null;
  let speechActive = false;
  let lastProcessedTranscript = '';

  const startWebSpeechAssist = () => {
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRec || speechActive) return;
    try {
      recognition = new SpeechRec();
      recognition.lang = 'ar-SA';
      recognition.continuous = true;
      recognition.interimResults = false;
      recognition.maxAlternatives = 1;

      recognition.onresult = (event: any) => {
        const api = window.__tartilApi;
        if (!api || !api.isMicOn()) return;

        const latestResult = event.results[event.results.length - 1];
        if (!latestResult || !latestResult.isFinal) return;

        const transcript = (latestResult[0]?.transcript || '').trim();
        if (!transcript || transcript === lastProcessedTranscript) return;
        lastProcessedTranscript = transcript;

        const incomingWords = transcript.split(/\s+/).filter(Boolean);
        stitchedWindowWords = overlapMerge(stitchedWindowWords.slice(-12), incomingWords);

        // Only confirm exact matched words; never flag mistakes from conversational Web Speech ASR
        const cue = api.getCueWord?.();
        if (!cue) return;
        const expectedWords = api.getAyahPlainWords?.(cue.surah, cue.ayah) ?? [];
        if (!expectedWords.length) return;

        const analysis = analyzeSpokenVsExpected({
          surah: cue.surah,
          ayah: cue.ayah,
          expectedWords,
          currentWordIndex: cue.wordIndex,
          spokenTranscript: incomingWords.slice(-2).join(' '),
        });
        if (analysis.matchedIndices.length > 0 && analysis.mistakes.length === 0) {
          api.confirmWords(cue.surah, cue.ayah, analysis.matchedIndices);
          updateScorePill();
        }
      };

      recognition.onend = () => {
        if (speechActive && window.__tartilApi?.isMicOn()) {
          try {
            recognition.start();
          } catch {
            // ignore restart error
          }
        }
      };

      recognition.start();
      speechActive = true;
    } catch {
      // Browser may not support SpeechRecognition
    }
  };

  const stopWebSpeechAssist = () => {
    speechActive = false;
    lastProcessedTranscript = '';
    stitchedWindowWords = [];
    if (recognition) {
      try {
        recognition.onend = null;
        recognition.stop();
      } catch {
        // ignore
      }
      recognition = null;
    }
  };

  // 6. Sync microphone state & live accuracy score pill (mistake detection is handled by settled Zipformer engine events)
  const micInterval = window.setInterval(() => {
    const api = window.__tartilApi;
    const isMic = api?.isMicOn() ?? false;
    if (isMic && !prevMicState) {
      // Starting a fresh microphone session clears previous session telemetry so results belong to this recitation
      sessionVerdictCache.clear();
      autoFlaggedKeys.clear();
      lastLiveAcousticTranscript = '';
    } else if (!isMic && prevMicState) {
      // Final flush when mic stops
      syncVerdictsFromEngine(true);
      persistCompletedSession();
    }
    prevMicState = isMic;
    if (isMic && !speechActive) {
      startWebSpeechAssist();
    } else if (!isMic && speechActive) {
      stopWebSpeechAssist();
    }
    updateScorePill();
  }, 350);

  return () => {
    window.clearInterval(micInterval);
    if (bannerHideTimer) window.clearTimeout(bannerHideTimer);
    if (activeAudio) {
      activeAudio.pause();
      activeAudio = null;
    }
    window.removeEventListener('tartil-mistake', onTartilMistakeEvent);
    window.removeEventListener('tartil-final-verdicts', onTartilFinalVerdicts);
    window.removeEventListener('tartil-tj-found', onTartilTjFound);
    window.removeEventListener('tartil-mistakes-cleared', onTartilMistakesCleared);
    window.removeEventListener('tartil-mistake-fixed', onTartilMistakeFixed);
    stopWebSpeechAssist();
    boostRow?.removeEventListener('click', onBoostRowClick);
    reciterRow?.removeEventListener('click', onReciterRowClick);
    slipHoldRow?.removeEventListener('click', onSlipHoldRowClick);
    barBoostBtn?.removeEventListener('click', onBarBoostClick);
    barHifzBtn?.removeEventListener('click', onBarHifzClick);
    settingsHifzSeg?.removeEventListener('click', onSettingsHifzClick);
    alertBanner?.remove();
    telemetrySheet?.remove();
    window.__onTartilRms = undefined;
  };
}
