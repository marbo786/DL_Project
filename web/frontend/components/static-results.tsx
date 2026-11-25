import type { ReactNode } from 'react';
import type { ResultsData } from '@/lib/results';

const percentage = (value: number) => (value * 100).toFixed(2) + '%';

function DataTable({ label, headings, children }: { label: string; headings: string[]; children: ReactNode }) {
  return <div className="static-data-scroll" role="region" aria-label={label} tabIndex={0} data-lenis-prevent>
    <table>
      <thead><tr>{headings.map(heading => <th key={heading} scope="col">{heading}</th>)}</tr></thead>
      <tbody>{children}</tbody>
    </table>
  </div>;
}

export function StaticResults({ data }: { data: ResultsData }) {
  const initial = data.risk[0];
  const previewThresholds = data.risk.filter((point, index) =>
    index === 0 || point.confidence === 0.55 || index === data.risk.length - 1);
  const reliabilityGroups = [
    { label: 'Detector', bins: data.reliability },
    { label: 'Constant prevalence', bins: data.constant_reliability },
  ];
  return <div className="static-results">
    <DataTable label="Published cohort results" headings={['Cohort', 'Receipts', 'AUROC', 'Macro F1', 'ECE']}>
      {data.cohorts.map(row => <tr key={row.name}>
        <th scope="row">{row.name}</th><td>{row.documents.toLocaleString('en-US')}</td>
        <td>{row.auroc.toFixed(4)}</td><td>{row.macro_f1.toFixed(4)}</td><td>{row.ece.toFixed(4)}</td>
      </tr>)}
    </DataTable>
    <p>Frozen retrospective results. Macro F1 uses the 0.50 binary cutoff. Lower expected calibration error (ECE) is better.</p>
    <details className="static-detail"><summary>Development confidence and review counts</summary>
      <p>At the lowest saved confidence threshold ({percentage(initial.confidence)}), {initial.review_count} of {data.development_documents} receipts met the frozen review cutoff, including {initial.edits_caught} edits and {initial.false_flags} false flags.</p>
      <DataTable label="Example saved confidence thresholds" headings={['Confidence', 'Coverage', 'Accepted error', 'Reviews', 'Edits', 'False flags']}>
        {previewThresholds.map(point => <tr key={point.confidence}>
          <th scope="row">{percentage(point.confidence)}</th><td>{percentage(point.coverage)}</td>
          <td>{point.accepted_error === null ? 'Undefined' : percentage(point.accepted_error)}</td>
          <td>{point.review_count}</td><td>{point.edits_caught}</td><td>{point.false_flags}</td>
        </tr>)}
      </DataTable>
      <a className="inline-link" href="/data/results.json">View all {data.risk.length} saved thresholds in the results JSON</a>
    </details>
    <details className="static-detail"><summary>Reliability bins and resource trade-offs</summary>
      <DataTable label="Reliability on the development split" headings={['Model', 'Receipts', 'Mean estimate', 'Observed edits']}>
        {reliabilityGroups.map(({ label, bins }) =>
          bins.map(bin => <tr key={label + '-' + bin.bin}>
            <th scope="row">{label}</th><td>{bin.count}</td>
            <td>{percentage(bin.mean_probability)}</td><td>{percentage(bin.observed_fraction)}</td>
          </tr>))}
      </DataTable>
      <p>Development recall at a fixed budget of {data.review_budget} reviews.</p>
      <DataTable label="Pareto trade-offs" headings={['Model', 'Size (MiB)', 'p95 (ms)', 'Recall']}>
        {data.pareto.map(row => <tr key={row.id}><th scope="row">{row.id}</th>
          <td>{row.checkpoint_mib.toFixed(2)}</td><td>{row.p95_ms.toFixed(1)}</td><td>{percentage(row.recall)}</td>
        </tr>)}
      </DataTable>
    </details>
    <details className="static-detail"><summary>Stress conditions and data sources</summary>
      <p>The same {data.stress_pairs} development pairs were tested across all conditions.</p>
      <DataTable label="Development stress results" headings={['Condition', 'AUROC', 'Edited recall', 'False-positive rate']}>
        {data.stress.map(row => <tr key={row.id}><th scope="row">{row.id}</th>
          <td>{row.auroc.toFixed(4)}</td><td>{percentage(row.recall)}</td><td>{percentage(row.fpr)}</td>
        </tr>)}
      </DataTable>
      <ul className="static-sources">{Object.values(data.sources).map(source => <li key={source.path}><code>{source.path}</code></li>)}</ul>
    </details>
  </div>;
}
