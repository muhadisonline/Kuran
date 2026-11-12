import {
  VOICE_BOOST_PRESETS,
  VoiceBoostMode,
} from './voiceDsp';
import {
  DetectedMistake,
  MistakeKind,
  ZipformerWordVerdict,
  analyzeSpokenVsExpected,
  evaluateVerdictMistakes,
  getReciterAyahAudioUrl,
  judgeAttemptTilawi,
  overlapMerge,
} from './mistakeDetector';

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
  retrySpot?: (surah: number, ayah: number, word?: number) => void;
  getMistakes?: () => TartilMistakeRecord[];
  clearMistakes?: () => void;
  getVerdicts?: () => ZipformerWordVerdict[];
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
    __tartilMistakeTester?: {
      simulateSpokenArabic: (spokenTranscript: string) => {
        matchedIndices: number[];
        mistakes: DetectedMistake[];
      };
      judgeAyahAttempt: (surah: number, ayah: number, heardWords: string[]) => ReturnType<typeof judgeAttemptTilawi>;
      getRecentAlerts: () => MistakeAlertItem[];
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

  // 2. Inject quick Voice Boost toggle button & Tilawi Accuracy Score Pill in the bottom bar (.t-bar)
  const bottomBar = rootEl.querySelector('.t-bar');
  let barBoostBtn: HTMLButtonElement | null = null;
  let barScorePill: HTMLSpanElement | null = null;

  if (bottomBar && !bottomBar.querySelector('.t-boost-pill')) {
    const pillWrap = document.createElement('div');
    pillWrap.className = 't-bar-pills';

    barBoostBtn = document.createElement('button');
    barBoostBtn.type = 'button';
    barBoostBtn.className = `t-boost-pill ${currentBoost === 'ultra' ? 'ultra' : ''}`;
    barBoostBtn.setAttribute('aria-label', 'Səs tutma gücü');
    barBoostBtn.title = 'Mikrofon səs tutma gücünü dəyişdir (Güclü / Çox güclü / Ultra)';

    barScorePill = document.createElement('span');
    barScorePill.className = 't-score-pill';
    barScorePill.title = 'Tilawi / Minhajy canlı qiraət dəqiqliyi';
    barScorePill.textContent = '100%';

    pillWrap.appendChild(barBoostBtn);
    pillWrap.appendChild(barScorePill);
    bottomBar.appendChild(pillWrap);
  }

  const updateScorePill = () => {
    if (!barScorePill) return;
    const st = window.__tartilApi?.getState?.();
    if (!st) return;
    const correct = Math.max(0, st.revealedCount);
    const errors = Math.max(0, st.mistakes);
    const total = correct + errors;
    const pct = total > 0 ? Math.round((100 * correct) / total) : 100;
    barScorePill.textContent = `${pct}%`;
    barScorePill.classList.toggle('warn', pct < 90 && pct >= 70);
    barScorePill.classList.toggle('bad', pct < 70);
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
        </div>
        <div class="t-alert-actions">
          <button type="button" class="t-alert-listen" title="Şeyx Husari / Əfasi tilavətini dinlə">🔊 Qarini dinlə</button>
          <button type="button" class="t-alert-retry">İndi düzəlt (Təkrar oxu)</button>
        </div>
      </div>
    `;
    alertBanner.hidden = false;

    const closeBtn = alertBanner.querySelector('.t-alert-close');
    closeBtn?.addEventListener('click', () => {
      if (alertBanner) alertBanner.hidden = true;
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
    }, 12000);
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

    const defaultExplanation =
      kind === 'possible_vowel'
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
      titleAz: detail.titleAz || detail.kindLabel || 'Tələffüz xətası',
      explanationAz: detail.note || defaultExplanation,
      expectedWord,
      heardWord: detail.heardWord,
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
    clearAlerts: () => {
      recentAlerts.length = 0;
      if (alertBanner) alertBanner.hidden = true;
      window.__tartilApi?.clearMistakes?.();
      updateScorePill();
    },
    playReciterAyah: (surah: number, ayah: number) => playReciterAyah(surah, ayah),
  };

  // 5. Parallel Browser Web Speech API (ar-SA) Assist — strictly confirms ONLY final exact-matching words, NEVER flags false mistakes on Qari Tajweed/Madd
  let recognition: any = null;
  let speechActive = false;
  let stitchedWindowWords: string[] = [];
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
    if (isMic && !speechActive) {
      startWebSpeechAssist();
    } else if (!isMic && speechActive) {
      stopWebSpeechAssist();
    }
    updateScorePill();
  }, 400);

  return () => {
    window.clearInterval(micInterval);
    if (bannerHideTimer) window.clearTimeout(bannerHideTimer);
    if (activeAudio) {
      activeAudio.pause();
      activeAudio = null;
    }
    window.removeEventListener('tartil-mistake', onTartilMistakeEvent);
    stopWebSpeechAssist();
    boostRow?.removeEventListener('click', onBoostRowClick);
    reciterRow?.removeEventListener('click', onReciterRowClick);
    slipHoldRow?.removeEventListener('click', onSlipHoldRowClick);
    barBoostBtn?.removeEventListener('click', onBarBoostClick);
    alertBanner?.remove();
    window.__onTartilRms = undefined;
  };
}
