'use client';

import { useRef, useState, type ReactNode } from 'react';
import { useInView, useReducedMotion } from 'framer-motion';
import { AnimatedNumber } from '@/components/animated-number';
import {
  Bar, BarChart, CartesianGrid, ComposedChart, LabelList, Legend, Line, LineChart, ReferenceDot, ReferenceLine,
  ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from 'recharts';
import type { ResultsData } from '@/lib/results';

const orange = 'var(--accent)';
const blue = 'var(--series)';
const text = 'var(--muted-foreground)';
const border = 'var(--border)';
const tick = { fill: text, fontSize: 11 };
const tip = { background: 'var(--card)', border: '1px solid var(--border)', color: 'var(--foreground)', borderRadius: '8px' };
const percent = (value: number) => Math.round(value * 100) + '%';
const precisePercent = (value: number) => (value * 100).toFixed(2) + '%';
const numberTip = (value: unknown) => typeof value === 'number' ? value.toFixed(4) : '';
const percentTip = (value: unknown) => typeof value === 'number' ? precisePercent(value) : 'Undefined';
const yTitle = (value: string) => ({ value, angle: -90, position: 'insideLeft' as const, fill: text, fontSize: 11 });
const xTitle = (value: string) => ({ value, position: 'insideBottom' as const, offset: -3, fill: text, fontSize: 11 });
const models: Record<string, { short: string; name: string }> = {
  'residual-fp32': { short: 'FP32', name: 'MobileNetV3 FP32' },
  'residual-int8': { short: 'INT8', name: 'MobileNetV3 INT8' },
  student: { short: 'PatchNet', name: 'Hard-label PatchNet' },
  distilled: { short: 'Distilled', name: 'Distilled PatchNet' },
};
const conditions: Record<string, string> = {
  clean: 'Clean', 'blur-1': 'Blur 0.7 px', 'blur-2': 'Blur 1.6 px',
  'brightness-1': 'Low light 0.85', 'brightness-2': 'Low light 0.65',
  'jpeg-1': 'JPEG 78', 'jpeg-2': 'JPEG 42', 'random-occlusion-1': 'Random cover 1',
  'random-occlusion-2': 'Random cover 2', 'edit-occlusion-1': 'Edit cover 1', 'edit-occlusion-2': 'Edit cover 2',
};

function Figure({ title, description, children }: { title: string; description: string; children: ReactNode }) {
  const ref = useRef<HTMLElement>(null);
  const inView = useInView(ref, { once: true, amount: .12 });
  const reduced = useReducedMotion();
  return (
    <figure ref={ref} className="chart-card" aria-label={title} data-chart-reveal={inView && !reduced ? 'active' : undefined}>
      <figcaption><h3>{title}</h3><p>{description}</p></figcaption>
      {children}
    </figure>
  );
}

export default function ResultsCharts({ data }: { data: ResultsData }) {
  const [point, setPoint] = useState(0);
  const [cost, setCost] = useState<'p95_ms' | 'checkpoint_mib'>('p95_ms');
  const [stress, setStress] = useState<'auroc' | 'recall' | 'fpr'>('auroc');
  const risk = data.risk[point];
  const costLabel = cost === 'p95_ms' ? 'p95 latency (ms)' : 'Checkpoint size (MiB)';
  const stressLabel = stress === 'auroc' ? 'AUROC' : stress === 'recall' ? 'Edited recall' : 'Genuine false-positive rate';
  const pareto = data.pareto.map(row => ({ ...row, ...models[row.id] }));
  const perturbations = data.stress.map(row => ({ ...row, condition: conditions[row.id] ?? row.id }));
  const reliability = [
    ...data.reliability.map(bin => ({ probability: bin.mean_probability, observed: bin.observed_fraction, count: bin.count })),
    ...data.constant_reliability.map(bin => ({ probability: bin.mean_probability, baseline: bin.observed_fraction, count: bin.count })),
  ].sort((a, b) => a.probability - b.probability);
  const comparison = (['auroc', 'macro_f1', 'ece'] as const).map((metric, index) => ({
    metric: ['AUROC', 'Macro F1', 'ECE'][index],
    id: data.cohorts[0][metric],
    shifted: data.cohorts[1][metric],
  }));

  return (
    <div className="charts-grid">
      <Figure title="In-domain and shifted results" description="Published retrospective results. Macro F1 uses the 0.50 binary cutoff. Lower ECE is better.">
        <dl className="metric-strip">{data.cohorts.map(cohort => <div key={cohort.name}>
          <dt>{cohort.name} AUROC</dt><dd><AnimatedNumber value={cohort.auroc} digits={4} /></dd>
        </div>)}</dl>
        <div className="chart-frame" role="group" aria-label="Bar chart of retrospective AUROC, macro F1, and ECE by cohort">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={comparison} margin={{ top: 12, right: 12, bottom: 20, left: 12 }} accessibilityLayer>
              <CartesianGrid vertical={false} stroke={border} />
              <XAxis dataKey="metric" tick={tick} label={xTitle('Metric')} /><YAxis domain={[0, 1]} tick={tick} label={yTitle('Score (0–1)')} />
              <Tooltip contentStyle={tip} formatter={numberTip} /><Legend verticalAlign="top" wrapperStyle={{ fontSize: 12 }} />
              <Bar dataKey="id" name={data.cohorts[0].name} fill={blue} isAnimationActive={false} />
              <Bar dataKey="shifted" name={data.cohorts[1].name} fill={orange} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <details className="chart-down"><summary>View cohort values</summary>
          <ul>{data.cohorts.map(row => <li key={row.name}>{row.name} ({row.documents.toLocaleString()} images) · AUROC {row.auroc.toFixed(4)} · F1 {row.macro_f1.toFixed(4)} · ECE {row.ece.toFixed(4)}</li>)}</ul>
        </details>
      </Figure>
      <Figure title="Reliability" description="Development bins only. Closer to the diagonal means predicted and observed rates agree.">
        <div className="chart-frame" role="group" aria-label="Observed edit rate versus estimated edit probability for development bins">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={reliability} margin={{ top: 12, right: 18, bottom: 22, left: 14 }} accessibilityLayer>
              <CartesianGrid stroke={border} />
              <XAxis type="number" dataKey="probability" domain={[0, 1]} tickFormatter={percent} tick={tick} label={xTitle('Estimated probability')} />
              <YAxis type="number" domain={[0, 1]} tickFormatter={percent} tick={tick} label={yTitle('Observed edit rate')} />
              <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={text} strokeDasharray="4 4" />
              <Tooltip contentStyle={tip} labelFormatter={value => precisePercent(Number(value))} formatter={percentTip} />
              <Legend verticalAlign="top" wrapperStyle={{ fontSize: 12 }} />
              <Scatter dataKey="observed" name="Fitted model" fill={orange} isAnimationActive={false} />
              <Scatter dataKey="baseline" name="Constant prevalence" fill={blue} shape="diamond" isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </Figure>
      <Figure title="Risk and coverage" description="Move between saved development thresholds. Risk is the error rate among accepted images.">
        <div className="chart-setting"><label htmlFor="review-confidence">Confidence threshold</label> <output aria-hidden="true">{precisePercent(risk.confidence)}</output>
          <input id="review-confidence" type="range" min={0} max={data.risk.length - 1} step={1} value={point}
            aria-valuetext={precisePercent(risk.confidence) + ' confidence'} onChange={event => setPoint(event.currentTarget.valueAsNumber)} /></div>
        <p className="chart-readout" aria-live="polite">{precisePercent(risk.coverage)} coverage · {risk.accepted_error === null ? 'error undefined at zero coverage' : precisePercent(risk.accepted_error) + ' accepted error'}</p>
        <dl className="metric-strip threshold-counts" aria-label="Development review outcomes">
          <div><dt>Sent to review</dt><dd><AnimatedNumber value={risk.review_count} /></dd></div>
          <div><dt>Edits caught</dt><dd><AnimatedNumber value={risk.edits_caught} /></dd></div>
          <div><dt>False flags</dt><dd><AnimatedNumber value={risk.false_flags} /></dd></div>
        </dl>
        <p className="threshold-note">Counts use the frozen nominal 5% review cutoff and the selected confidence gate on {data.development_documents} saved development scores. Coverage is the share above the confidence gate.</p>
        <div className="chart-frame" role="group" aria-label="Coverage versus accepted error across saved development thresholds">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart margin={{ top: 12, right: 18, bottom: 22, left: 14 }} data={data.risk} accessibilityLayer>
              <CartesianGrid stroke={border} />
              <XAxis dataKey="coverage" type="number" domain={[0, 1]} tickFormatter={percent} tick={tick} label={xTitle('Coverage')} />
              <YAxis domain={[0, 1]} tickFormatter={percent} tick={tick} label={yTitle('Accepted error')} />
              <Tooltip contentStyle={tip} labelFormatter={value => 'Coverage ' + precisePercent(Number(value))} formatter={percentTip} />
              <Legend verticalAlign="top" wrapperStyle={{ fontSize: 12 }} />
              <Line type="linear" dataKey="accepted_error" name="Accepted error" stroke={orange} dot={false} strokeWidth={2} isAnimationActive={false} />
              {risk.accepted_error !== null &&
                <ReferenceDot x={risk.coverage} y={risk.accepted_error} r={5} fill={blue} stroke={text} />}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Figure>
      <Figure title="Pareto trade-offs" description={'Development recall at a budget of ' + data.review_budget + ' reviews. All designs found the same share of edits.'}>
        <label className="chart-setting">Cost axis
          <select value={cost} onChange={event => setCost(event.currentTarget.value as typeof cost)}>
            <option value="p95_ms">p95 latency</option><option value="checkpoint_mib">Checkpoint size</option>
          </select>
        </label>
        <div className="chart-frame" role="group" aria-label={'Recall versus ' + costLabel + ' at the same review budget'}>
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 12, right: 28, bottom: 22, left: 14 }} accessibilityLayer>
              <CartesianGrid stroke={border} />
              <XAxis dataKey={cost} name={costLabel} type="number" tick={tick} label={xTitle(costLabel)} domain={[0, 'auto']} />
              <YAxis dataKey="recall" name="Recall" type="number" tick={tick} tickFormatter={percent} domain={[0, .2]} label={yTitle('Edited recall')} />
              <ZAxis range={[65, 65]} />
              <Tooltip contentStyle={tip} labelFormatter={(_, payload) => payload?.[0]?.payload?.name ?? 'Model'}
                formatter={(value, name) => typeof value === 'number' && name === 'Recall'
                  ? precisePercent(value) : typeof value === 'number' ? value.toFixed(cost === 'p95_ms' ? 2 : 4) : ''} />
              <Legend verticalAlign="top" wrapperStyle={{ fontSize: 12 }} />
              {pareto.map(row => <Scatter key={row.id} name={row.name} data={[row]} fill={row.id.startsWith('residual') ? orange : blue}
                shape={row.id === 'distilled' ? 'diamond' : 'circle'} isAnimationActive={false}>
                  <LabelList dataKey="short" position={row.id === 'distilled' ? 'bottom' : 'top'} fill={text} fontSize={10} />
                </Scatter>)}
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </Figure>
      <Figure title="Stress-test degradation" description={'The same ' + data.stress_pairs + ' development pairs under blur, low light, JPEG, and occlusion. The dashed line marks the clean baseline.'}>
        <label className="chart-setting">Metric
          <select value={stress} onChange={event => setStress(event.currentTarget.value as typeof stress)}>
            <option value="auroc">AUROC</option><option value="recall">Edited recall</option><option value="fpr">False-positive rate</option>
          </select>
        </label>
        <div className="chart-frame chart-tall" role="group" aria-label={'Stress-test ' + stressLabel + ' by condition. Dashed line indicates clean baseline.'}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={perturbations} layout="vertical" margin={{ top: 10, right: 22, bottom: 28, left: 8 }} accessibilityLayer>
              <CartesianGrid horizontal={false} stroke={border} />
              <XAxis type="number" domain={[0, 1]} tickFormatter={stress === 'auroc' ? value => value.toFixed(2) : percent}
                tick={tick} label={xTitle(stress === 'auroc' ? 'AUROC (0–1)' : stressLabel + ' (%)')} />
              <YAxis dataKey="condition" type="category" width={94} tick={tick} />
              <ReferenceLine x={data.stress[0][stress]} stroke={blue} strokeDasharray="3 3" />
              <Tooltip contentStyle={tip} formatter={stress === 'auroc' ? numberTip : percentTip} />
              <Legend verticalAlign="top" wrapperStyle={{ fontSize: 12 }} />
              <Bar dataKey={stress} name={stressLabel} fill={orange} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Figure>
    </div>
  );
}
