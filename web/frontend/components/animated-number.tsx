'use client';

import { animate, useInView, useMotionValue, useMotionValueEvent, useReducedMotion } from 'framer-motion';
import { useEffect, useRef, useState } from 'react';

export function AnimatedNumber({ value, digits = 0, suffix = '' }: {
  value: number; digits?: number; suffix?: string;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const visible = useInView(ref, { once: true });
  const reduced = useReducedMotion();
  const number = useMotionValue(0);
  const [display, setDisplay] = useState(value);
  useMotionValueEvent(number, 'change', setDisplay);
  useEffect(() => {
    if (!visible) return;
    if (reduced) {
      number.set(value);
      return;
    }
    const duration = Number.parseFloat(ref.current ? getComputedStyle(ref.current).getPropertyValue('--motion-value') : '') / 1000;
    const animation = animate(number, value, { duration: Number.isFinite(duration) ? duration : .32, ease: [.2, .7, .3, 1] });
    return () => animation.stop();
  }, [number, reduced, value, visible]);
  return <span ref={ref} className="animated-number">
    <span className="sr-only">{value.toFixed(digits)}{suffix}</span>
    <span aria-hidden="true">{(visible && !reduced ? display : value).toFixed(digits)}{suffix}</span>
  </span>;
}
