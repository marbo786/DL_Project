import { EvidenceSpecimen, type SavedEvidence } from '@/components/evidence-specimen';
import { Workbench } from '@/components/workbench';
import samples from '@/data/sample-results.json';
import results from '@/data/results.json';

function savedEvidence(id: 'edited' | 'original'): SavedEvidence {
  const { decision, calibrated_probability, confidence, top_tile, tiles } = samples[id].result;
  if (decision !== 'prioritize_review' && decision !== 'normal_queue' && decision !== 'inconclusive') {
    throw new Error('Unrecognized saved receipt decision');
  }
  return { decision, calibrated_probability, confidence, top_tile, tiles };
}

export default function Page() {
  const evidence = { edited: savedEvidence('edited'), original: savedEvidence('original') };
  return <Workbench data={results} evidence={<EvidenceSpecimen evidence={evidence} />} />;
}
