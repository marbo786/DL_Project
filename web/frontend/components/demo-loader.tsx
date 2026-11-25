'use client';

import { useEffect, useRef, useState, type ComponentType } from 'react';

export function DemoLoader() {
  const ref = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [failed, setFailed] = useState(false);
  const [Demo, setDemo] = useState<ComponentType | null>(null);

  useEffect(() => {
    if (open || !ref.current) return;
    if (!('IntersectionObserver' in window)) { setOpen(true); return; }
    const observer = new IntersectionObserver(entries => {
      if (entries.some(entry => entry.isIntersecting)) setOpen(true);
    }, { rootMargin: '48px' });
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, [open]);

  useEffect(() => {
    if (!open) return;
    let active = true;
    void import('@/components/demo').then(module => {
      if (active) setDemo(() => module.default);
    }).catch(() => { if (active) setFailed(true); });
    return () => { active = false; };
  }, [open]);

  return <div ref={ref} className="studio-reserve">
    {Demo ? <Demo /> : <div className="section-status" data-reserve="studio">
      <p>{failed ? 'The receipt studio could not load.' : 'The receipt studio supports saved samples and local image previews.'}</p>
      {failed && <button type="button" className="control control-primary" onClick={() => window.location.reload()}>Reload page</button>}
      <noscript>The interactive studio requires JavaScript. The research results below are available on this page.</noscript>
    </div>}
  </div>;
}
