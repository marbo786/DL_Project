'use client';

import { FileCheck2, Grid2X2, ListFilter, ScanLine, SlidersHorizontal } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { steps } from '@/lib/pipeline-steps';
import receipt from './assets/receipt-hero.webp';
import specimen from '@/data/sample-results.json';
import { decisionLabels, percent, type Decision } from '@/lib/receipt-client';

const icons = [FileCheck2, Grid2X2, ScanLine, SlidersHorizontal, ListFilter];

export default function Pipeline() {
  const [step, setStep] = useState(0);
  const track = useRef<HTMLDivElement>(null);
  const panel = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const trackElement = track.current;
    const panelElement = panel.current;
    if (!trackElement || !panelElement) return;
    const reduced = matchMedia('(prefers-reduced-motion: reduce), (max-height: 540px)');
    let pending = 0;
    function update() {
      if (!trackElement || !panelElement || pending || reduced.matches) return;
      pending = requestAnimationFrame(() => {
        pending = 0;
        const bounds = trackElement.getBoundingClientRect();
        const offset = parseFloat(getComputedStyle(panelElement).top);
        const distance = bounds.height - panelElement.offsetHeight;
        const progress = (offset - bounds.top) / distance;
        if (distance <= 0 || progress < 0 || progress > 1) return;
        setStep(Math.min(steps.length - 1, Math.floor(progress * steps.length)));
      });
    }
    window.addEventListener('scroll', update, { passive: true });
    update();
    return () => {
      cancelAnimationFrame(pending);
      window.removeEventListener('scroll', update);
    };
  }, []);
  const selected = steps[step];
  const { x, y, width, height } = specimen.edited.result.top_tile;
  return (
    <div className="pipeline-track" ref={track}>
     <div className="pipeline" ref={panel}>
      <ol className="pipeline-steps">
        {steps.map(({ title }, index) => {
          const Icon = icons[index];
          return <li key={title}>
          <button type="button" aria-pressed={index === step} aria-controls="pipeline-detail" onClick={() => setStep(index)}>
            <Icon size={19} aria-hidden="true" />{title}
          </button>
        </li>;
        })}
      </ol>
      <div id="pipeline-detail" className="pipeline-detail">
        <div className="pipeline-preview" data-step={step} aria-hidden="true">
          <img src={receipt.src} width={receipt.width} height={receipt.height} alt="" loading="lazy" decoding="async" />
          {step >= 1 && step <= 2 && <svg viewBox="0 0 1 1" preserveAspectRatio="none">
            {specimen.edited.result.tiles.map((tile, index) => <rect key={index} x={tile.x} y={tile.y}
              width={tile.width} height={tile.height} fill="none" stroke="#56432f" vectorEffect="non-scaling-stroke" />)}
            {step === 2 && <rect x={x} y={y} width={width} height={height} fill="#ebbd8780"
              stroke="#79431f" strokeWidth={2} vectorEffect="non-scaling-stroke" />}
          </svg>}
        </div>
        <div className="pipeline-narration">
          <label className="pipeline-progress">Step {step + 1} of {steps.length}
            <progress max={steps.length} value={step + 1} /></label>
          <div key={selected.title} className="pipeline-copy">
            <h3>{selected.title}</h3><p>{selected.text}</p>
            {step === 3 && <p className="pipeline-value">{percent(specimen.edited.result.calibrated_probability)} estimated edit probability</p>}
            {step === 4 && <p className="pipeline-value">Saved decision {decisionLabels[specimen.edited.result.decision as Decision]}</p>}
          </div>
        </div>
      </div>
     </div>
    </div>
  );
}
