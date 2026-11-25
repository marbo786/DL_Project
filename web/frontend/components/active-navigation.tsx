'use client';

import { useEffect, useState } from 'react';

const links = [['demo', 'Demo'], ['results', 'Results'], ['method', 'Method'], ['limits', 'Limits']] as const;

export function ActiveNavigation() {
  const [current, setCurrent] = useState<string | undefined>();

  function update() {
    const header = document.querySelector('[data-scroll-header]')?.getBoundingClientRect().bottom ?? 0;
    let section;
    for (const [id] of links) {
      const top = document.querySelector('#' + id + ' [data-scroll-heading]')?.getBoundingClientRect().top;
      if (top !== undefined && top <= header + 28) section = id;
    }
    setCurrent(section);
  }

  useEffect(() => {
    let pending = 0;
    function schedule() {
      if (!pending) pending = requestAnimationFrame(() => { pending = 0; update(); });
    }
    const observer = new ResizeObserver(schedule);
    window.addEventListener('scroll', schedule, { passive: true });
    observer.observe(document.body);
    update();
    return () => {
      observer.disconnect();
      cancelAnimationFrame(pending);
      window.removeEventListener('scroll', schedule);
    };
  }, []);

  return <nav aria-label="Main navigation">{links.map(([id, label]) =>
    <a key={id} href={'#' + id} aria-current={current === id ? 'location' : undefined}>{label}</a>,
  )}</nav>;
}
