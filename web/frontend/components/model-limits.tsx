import { Check, CircleAlert } from 'lucide-react';
import type { ResultsData } from '@/lib/results';

const repository = 'https://github.com/marbo786/DL_Project';

export function ModelLimits({ data }: { data: ResultsData }) {
  const formatPercent = (value: number) => (value * 100).toLocaleString(undefined, { maximumFractionDigits: 2 }) + '%';
  const rows = [
    { label: 'WildReceipt priority recall', ...data.limits.priority_recall, format: formatPercent, atLeast: true },
    { label: 'WildReceipt macro F1', ...data.limits.shifted_f1, format: (value: number) => value.toFixed(4), atLeast: true },
    { label: 'Development coverage', ...data.limits.coverage, format: formatPercent, atLeast: true },
    { label: 'Development accepted error', ...data.limits.accepted_error, format: formatPercent, atLeast: false },
    { label: 'Estimated reviews per 1,000', ...data.limits.queue_load, format: (value: number) => value.toLocaleString(), atLeast: false },
    { label: 'WildReceipt ECE', ...data.limits.shifted_ece, format: (value: number) => value.toFixed(4), atLeast: false },
    { label: 'Development CPU median', ...data.limits.median_ms, format: (value: number) => value + ' ms', atLeast: false },
    { label: 'Model parameters', ...data.limits.parameters, format: (value: number) => value.toLocaleString(), atLeast: false },
  ];
  return (
    <div className="limits-list">
      <details open><summary>Which targets did the model miss?</summary>
        <div className="limits-table-wrap" tabIndex={0} role="region" aria-label="Measured outcomes and targets" data-lenis-prevent><table className="limits-table">
          <thead><tr><th scope="col">Measure</th><th scope="col">Observed</th><th scope="col">Target</th><th scope="col">Outcome</th></tr></thead>
          <tbody>{rows.map(row => {
            const passed = row.observed !== null && (row.atLeast ? row.observed >= row.target : row.observed <= row.target);
            return <tr key={row.label}>
              <th scope="row">{row.label}</th><td>{row.observed === null ? 'Undefined' : row.format(row.observed)}</td>
              <td>{row.atLeast ? '≥' : '≤'} {row.format(row.target)}</td>
              <td>{passed ? <Check size={15} aria-hidden="true" /> : <CircleAlert size={15} aria-hidden="true" />}{row.observed === null ? 'N/A' : passed ? 'Met' : 'Missed'}</td>
            </tr>;
          })}</tbody>
        </table></div>
        <p>Before deferral, the frozen cutoff on WildReceipt found {formatPercent(data.published_shift.binary_recall)} of edits at {formatPercent(data.published_shift.binary_fpr)} genuine false positives. The confidence gate covered {formatPercent(data.published_shift.coverage)} with {formatPercent(data.published_shift.accepted_error)} accepted error. Estimated queue load assumes 1% edit prevalence.</p>
        <a className="inline-link" href={repository + '/blob/main/docs/ForgeryLens_Technical_Report.pdf'}>Read the published course report</a>
      </details>
      <details><summary>Intended use and false accusations</summary>
        <p>Use this course prototype to inspect signs of visual edits. A wrong flag can delay a legitimate payment or draw unnecessary suspicion. Confirm the claim against transaction records before escalating. An inconclusive result calls for standard manual review.</p>
      </details>
      <details><summary>How uploads are handled</summary>
        <p>The browser keeps previews in this session. Saved-sample mode sends no receipt to an API. Live mode sends the selected image to the configured research service, which scores it in memory. Use only images you’re authorized to share.</p>
      </details>
      <details><summary>Model, data, and credits</summary>
        <p>The frozen detector is based on TorchVision MobileNetV3-Small ImageNet V1 weights. TorchVision code uses BSD 3-Clause. The exact checkpoint’s separate redistribution terms remain unresolved. The receipt study uses AIForge-Doc v2.0.1 data, with CORD, SROIE, and WildReceipt as source corpora.</p>
        <div className="credit-links">
          <a className="inline-link" href="https://docs.pytorch.org/vision/main/models.html#general-information-on-pre-trained-weights">TorchVision weights notice</a>
          <a className="inline-link" href="https://huggingface.co/datasets/Scam-AI/AIForge-Doc-v2">AIForge-Doc dataset card</a>
        </div>
      </details>
    </div>
  );
}

