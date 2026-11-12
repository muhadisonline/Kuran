/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useRef } from 'react';
import { attachHybridVoiceBooster } from './audio/hybridBooster';

declare global {
  interface Window {
    __mountTartil?: (el: HTMLElement, opts?: { hash?: boolean }) => () => void;
  }
}

export default function App() {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    let cleanupEngine: (() => void) | undefined;
    let cleanupBooster: (() => void) | undefined;
    let mounted = false;

    const init = () => {
      if (mounted || !el || typeof window.__mountTartil !== 'function') return;
      mounted = true;
      cleanupEngine = window.__mountTartil(el, { hash: true });
      cleanupBooster = attachHybridVoiceBooster(el);
    };

    if (typeof window.__mountTartil === 'function') {
      init();
    } else {
      window.addEventListener('tartil-engine-ready', init, { once: true });
    }

    return () => {
      window.removeEventListener('tartil-engine-ready', init);
      cleanupBooster?.();
      cleanupEngine?.();
    };
  }, []);

  return <div id="app" ref={containerRef} />;
}
