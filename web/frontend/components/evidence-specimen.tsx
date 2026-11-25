'use client';

import { useState, type CSSProperties, type PointerEvent } from 'react';
import { decisionLabels, percent, type Analysis, type Bounds } from '@/lib/receipt-client';
import specimen from '../../public/specimen.json';
import editedImage from './assets/receipt-hero.webp';
import originalImage from './assets/receipt-original-hero.webp';

const specimens = [
  { id: 'edited', label: 'Edited', image: editedImage },
  { id: 'original', label: 'Original', image: originalImage },
] as const;

type SpecimenId = (typeof specimens)[number]['id'];
export type SavedEvidence = Pick<Analysis, 'decision' | 'calibrated_probability' | 'confidence' | 'top_tile' | 'tiles'>;

function boxStyle({ x, y, width, height }: Bounds): CSSProperties {
  return { left: x * 100 + '%', top: y * 100 + '%', width: width * 100 + '%', height: height * 100 + '%' };
}

const sameOrigin = (a: Bounds, b: Bounds) => a.x === b.x && a.y === b.y;

export function EvidenceSpecimen({ evidence }: { evidence: Record<SpecimenId, SavedEvidence> }) {
  const [split, setSplit] = useState(100);
  const [scan, setScan] = useState(0);
  const [hover, setHover] = useState<{ x: number; y: number } | null>(null);
  const selected = split === 0 ? 'original' : 'edited';
  const underPointer = hover && split < 100 && hover.x >= split ? 'original' : selected;
  const result = evidence[selected];
  const active = evidence[underPointer];
  const hoveredTile = hover ? active.tiles?.find(tile =>
    hover.x >= tile.x * 100 && hover.x < (tile.x + tile.width) * 100 &&
    hover.y >= tile.y * 100 && hover.y < (tile.y + tile.height) * 100) : undefined;
  const topTile = result.tiles?.find(tile => sameOrigin(tile, result.top_tile));

  function track(event: PointerEvent<HTMLDivElement>) {
    if (event.pointerType !== 'mouse') return;
    const bounds = event.currentTarget.getBoundingClientRect();
    setHover({ x: 100 * (event.clientX - bounds.left) / bounds.width,
      y: 100 * (event.clientY - bounds.top) / bounds.height });
  }

  return (
    <aside className="evidence-panel" aria-labelledby="evidence-title">
      <div className="evidence-header">
        <h2 id="evidence-title">Fictional receipt</h2>
        <div className="evidence-select" role="group" aria-label="Receipt specimen">
          {specimens.map(specimen => (
            <button key={specimen.id} type="button" aria-label={specimen.label + ' specimen'}
              aria-pressed={selected === specimen.id && (split === 0 || split === 100)}
              onClick={() => { setSplit(specimen.id === 'edited' ? 100 : 0); setHover(null); }}>
              {specimen.label}
            </button>
          ))}
        </div>
      </div>
      <div className="evidence-body">
        <figure className="evidence-figure">
          <div className="evidence-paper" onPointerMove={track} onPointerLeave={() => setHover(null)}
            role="img" aria-label={split === 0 ? 'Original fictional Paper Trail receipt' : split === 100
              ? 'Edited fictional Paper Trail receipt' : 'Split view of the edited and original fictional receipts'}>
            {split < 100 && <img src={originalImage.src} width={originalImage.width} height={originalImage.height} alt="" />}
            <div className="specimen-edited-layer" style={{ clipPath: 'inset(0 ' + (100 - split) + '% 0 0)' }}>
              <img src={editedImage.src} width={editedImage.width} height={editedImage.height} alt="" fetchPriority="high" />
              {split < 100 && <span className="specimen-edit-region" aria-hidden="true" style={boxStyle(specimen.edit_region)} />}
              <span className="evidence-tile" aria-hidden="true" style={boxStyle(evidence.edited.top_tile)} />
            </div>
            {split === 0 && <span className="evidence-tile" aria-hidden="true" style={boxStyle(result.top_tile)} />}
            {split > 0 && split < 100 && <span className="specimen-divider" aria-hidden="true" style={{ left: split + '%' }} />}
            {hoveredTile && <span className="specimen-hover-tile" aria-hidden="true" style={boxStyle(hoveredTile)} />}
            {hover && <span className="specimen-loupe" aria-hidden="true" style={{
              left: hover.x + '%', top: hover.y + '%',
              backgroundImage: 'url(' + (underPointer === 'edited' ? editedImage.src : originalImage.src) + ')',
              backgroundPosition: hover.x + '% ' + hover.y + '%',
            }} />}
            {scan > 0 && <span key={scan} className="specimen-sweep" aria-hidden="true" />}
          </div>
          <figcaption><span aria-hidden="true" /> Highest-scoring tile{split > 0 && split < 100 && ' · Dashed outline marks the known edit'}</figcaption>
          <label className="specimen-range">Compare original and edited
            <input type="range" min="0" max="100" value={split} aria-valuetext={split + '% edited visible'}
              onChange={event => { setSplit(event.currentTarget.valueAsNumber); setHover(null); }} />
          </label>
        </figure>
        <div className="evidence-readout" aria-live="polite">
          <p className="evidence-source">Saved model output</p>
          <p className="decision" data-decision={result.decision}>{decisionLabels[result.decision]}</p>
          <dl>
            <div><dt>Estimated edit probability</dt><dd>{percent(result.calibrated_probability)}</dd></div>
            <div><dt>Prediction confidence</dt><dd>{percent(result.confidence)}</dd></div>
          </dl>
          <p className="specimen-score">{hoveredTile ? 'Hovered tile' : 'Highest tile'} logit
            <strong>{(hoveredTile ?? topTile)?.logit.toFixed(3) ?? 'Unavailable'}</strong>
          </p>
          {split > 0 && split < 100 && <p className="specimen-score-source">Readout for the edited specimen</p>}
          <button type="button" className="specimen-scan-button" onClick={() => setScan(value => value + 1)}>Replay scan</button>
        </div>
      </div>
    </aside>
  );
}
