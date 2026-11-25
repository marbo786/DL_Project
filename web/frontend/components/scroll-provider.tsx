'use client';

import type Lenis from 'lenis';
import { useEffect, type ReactNode } from 'react';

const anchorGap = 12;
const scrollKeys = new Set(['ArrowUp', 'ArrowDown', 'PageUp', 'PageDown', 'Home', 'End', ' ', 'Tab']);

type Anchor = {
  target: HTMLElement;
  focus: boolean;
  moving: boolean;
  destination: number | null;
};

function hashTarget(hash: string) {
  if (!hash || hash === '#') return null;
  try {
    return document.getElementById(decodeURIComponent(hash.slice(1)));
  } catch {
    return null;
  }
}

export function ScrollProvider({ children }: { children: ReactNode }) {
  useEffect(() => {
    const header = document.querySelector<HTMLElement>('[data-scroll-header]');
    const root = document.documentElement;
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    let lenis: Lenis | null = null;
    let anchor: Anchor | null = null;
    let restorePosition: number | null = null;
    let headerOffset = anchorGap;
    let syncFrame = 0;
    let motionRequest = 0;

    function position(current: Anchor) {
      const target = current.target;
      const heading = target.tagName === 'SECTION'
        ? target.querySelector<HTMLElement>('[data-scroll-heading]') ?? target
        : target;
      const top = window.scrollY + heading.getBoundingClientRect().top - headerOffset;
      const max = Math.max(0, (document.scrollingElement?.scrollHeight ?? root.scrollHeight) - window.innerHeight);
      return { heading, top: Math.max(0, Math.min(max, top)) };
    }

    function align(current: Anchor, animate: boolean) {
      const { heading, top } = position(current);
      const smooth = animate && !!lenis && !reducedMotion.matches;
      if (smooth && current.moving && current.destination !== null && Math.abs(current.destination - top) < 1) return;
      current.destination = top;
      current.moving = smooth && Math.abs(window.scrollY - top) >= 1;

      function finish() {
        if (anchor !== current) return;
        current.moving = false;
        if (current.focus) {
          current.focus = false;
          heading.focus({ preventScroll: true });
        }
        if (Math.abs(position(current).top - window.scrollY) >= 1) scheduleSync();
      }

      if (lenis) {
        lenis.scrollTo(top, {
          immediate: !current.moving,
          duration: 0.65,
          onComplete: finish,
        });
      } else {
        window.scrollTo({ top, behavior: 'instant' });
        finish();
      }
    }

    function scheduleSync() {
      if (syncFrame) return;
      syncFrame = window.requestAnimationFrame(() => {
        syncFrame = 0;
        headerOffset = Math.ceil(header?.getBoundingClientRect().height ?? 0) + anchorGap;
        root.style.setProperty('--scroll-header-offset', headerOffset + 'px');
        if (anchor) align(anchor, anchor.moving);
        else if (restorePosition !== null) {
          window.scrollTo({ top: restorePosition, behavior: 'instant' });
          restorePosition = null;
        }
      });
    }

    function go(target: HTMLElement, focus: boolean, animate: boolean) {
      restorePosition = null;
      anchor = { target, focus, moving: false, destination: null };
      headerOffset = Math.ceil(header?.getBoundingClientRect().height ?? 0) + anchorGap;
      align(anchor, animate);
    }

    function releaseAnchor() {
      const moving = anchor?.moving;
      anchor = null;
      restorePosition = null;
      if (moving && lenis) {
        lenis.stop();
        lenis.start();
      }
    }

    function handleClick(event: MouseEvent) {
      if (event.defaultPrevented || event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      const link = event.target instanceof Element ? event.target.closest<HTMLAnchorElement>('a[href^="#"]') : null;
      if (!link || link.hasAttribute('download') || (link.target && link.target !== '_self')) return;
      const target = hashTarget(link.hash);
      if (!target) return;
      event.preventDefault();
      if (window.location.hash !== link.hash) {
        if (!window.location.hash) history.replaceState({ ...history.state, forgeryLensScrollY: window.scrollY }, '');
        history.pushState(null, '', link.hash);
      }
      go(target, true, !link.classList.contains('skip-link'));
    }

    function handleHistory() {
      const target = hashTarget(window.location.hash);
      if (!target) {
        releaseAnchor();
        const saved = history.state?.forgeryLensScrollY;
        if (!window.location.hash && typeof saved === 'number') {
          restorePosition = saved;
          scheduleSync();
        }
        return;
      }
      if (!anchor || anchor.target !== target) anchor = { target, focus: false, moving: false, destination: null };
      scheduleSync();
    }

    function handleKey(event: KeyboardEvent) {
      if (scrollKeys.has(event.key)) releaseAnchor();
    }

    function destroyLenis() {
      lenis?.destroy();
      lenis = null;
    }

    function setMotionPreference() {
      const request = ++motionRequest;
      if (reducedMotion.matches) {
        destroyLenis();
        if (anchor) align(anchor, false);
        return;
      }
      if (lenis) return;
      void import('lenis').then(({ default: Scroll }) => {
        if (request !== motionRequest) return;
        lenis = new Scroll({ autoRaf: true, lerp: 0.13, syncTouch: false, anchors: false });
      }).catch(() => {});
    }

    const observer = new ResizeObserver(scheduleSync);
    observer.observe(document.body);
    if (header) observer.observe(header);
    document.addEventListener('click', handleClick);
    window.addEventListener('hashchange', handleHistory);
    window.addEventListener('popstate', handleHistory);
    window.addEventListener('resize', scheduleSync);
    window.addEventListener('pointerdown', releaseAnchor, { passive: true });
    window.addEventListener('touchstart', releaseAnchor, { passive: true });
    window.addEventListener('wheel', releaseAnchor, { passive: true, capture: true });
    window.addEventListener('keydown', handleKey);
    reducedMotion.addEventListener('change', setMotionPreference);
    const initial = hashTarget(window.location.hash);
    if (initial) go(initial, false, false);
    scheduleSync();
    setMotionPreference();

    return () => {
      motionRequest++;
      destroyLenis();
      window.cancelAnimationFrame(syncFrame);
      root.style.removeProperty('--scroll-header-offset');
      observer.disconnect();
      document.removeEventListener('click', handleClick);
      window.removeEventListener('hashchange', handleHistory);
      window.removeEventListener('popstate', handleHistory);
      window.removeEventListener('resize', scheduleSync);
      window.removeEventListener('pointerdown', releaseAnchor);
      window.removeEventListener('touchstart', releaseAnchor);
      window.removeEventListener('wheel', releaseAnchor, true);
      window.removeEventListener('keydown', handleKey);
      reducedMotion.removeEventListener('change', setMotionPreference);
    };
  }, []);

  return children;
}
