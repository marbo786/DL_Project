'use client';

import { useState, type ComponentType } from 'react';
import { readResults, type ResultsData } from '@/lib/results';

export function ChartExplorer() {
  const [charts, setCharts] = useState<{ View: ComponentType<{ data: ResultsData }>; data: ResultsData } | null>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'failed'>('idle');

  async function open() {
    setStatus('loading');
    try {
      const [module, data] = await Promise.all([import('@/components/results-charts'), readResults()]);
      setCharts({ View: module.default, data });
      setStatus('idle');
    } catch {
      setStatus('failed');
    }
  }

  return <div className="chart-explorer">
    <button type="button" className="control control-outline" aria-expanded={!!charts}
      aria-controls="interactive-charts" aria-disabled={status === 'loading'}
      onClick={() => {
        if (charts) setCharts(null);
        else if (status !== 'loading') void open();
      }}>
      {charts ? 'Close interactive charts' : status === 'failed' ? 'Retry interactive charts' : 'Explore interactive charts'}
    </button>
    <div id="interactive-charts" role={charts ? undefined : 'status'}>
      {charts && <charts.View data={charts.data} />}
      {status === 'loading' && <p>Loading interactive charts…</p>}
      {status === 'failed' && <p>The chart controls could not load. The saved results are available above.</p>}
    </div>
  </div>;
}
