'use client';

import { Moon, Sun } from 'lucide-react';
import { useEffect, useState } from 'react';

export function ThemeToggle() {
  const [light, setLight] = useState(false);
  useEffect(() => setLight(document.documentElement.dataset.theme === 'light'), []);

  function toggle() {
    const theme = light ? 'dark' : 'light';
    document.documentElement.dataset.theme = theme;
    setLight(!light);
    try { localStorage.setItem('forgerylens-theme', theme); } catch {}
  }

  return <button type="button" className="control control-outline theme-toggle" onClick={toggle}>
    {light ? <Moon size={17} aria-hidden="true" /> : <Sun size={17} aria-hidden="true" />}
    {light ? 'Dark mode' : 'Light mode'}
  </button>;
}
