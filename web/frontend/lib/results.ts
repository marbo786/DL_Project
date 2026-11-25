export type Limit = { observed: number | null; target: number };
export type ResultsData = {
  sources: Record<string, { path: string; sha256: string }>;
  cohorts: { name: string; auroc: number; macro_f1: number; ece: number; documents: number }[];
  reliability: { bin: number; count: number; mean_probability: number; observed_fraction: number }[];
  constant_reliability: ResultsData['reliability'];
  risk: { confidence: number; coverage: number; accepted_error: number | null;
    review_count: number; edits_caught: number; false_flags: number }[];
  development_documents: number;
  pareto: { id: string; checkpoint_mib: number; median_ms: number; p95_ms: number; recall: number }[];
  review_budget: number;
  stress: { id: string; auroc: number; recall: number; fpr: number }[];
  stress_pairs: number;
  limits: Record<'priority_recall' | 'shifted_f1' | 'coverage' | 'accepted_error' | 'queue_load' | 'shifted_ece' | 'median_ms' | 'parameters', Limit>;
  published_shift: { coverage: number; accepted_error: number; binary_fpr: number; binary_recall: number };
};

export async function readResults(signal?: AbortSignal): Promise<ResultsData> {
  const response = await fetch('/data/results.json', { signal });
  if (!response.ok) throw new Error('The saved results could not be loaded.');
  const data = await response.json() as ResultsData;
  if (!data.sources || !data.limits || !data.published_shift || ![data.cohorts, data.reliability, data.risk, data.pareto, data.stress]
    .every(Array.isArray)) throw new Error('The saved results file is unreadable.');
  return data;
}
