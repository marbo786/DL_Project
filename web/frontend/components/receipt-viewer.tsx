'use client';

import { Grid2X2, Minus, Plus, RotateCcw } from 'lucide-react';
import { useRef, useState } from 'react';
import type { Analysis } from '@/lib/receipt-client';

export function ReceiptViewer({ src, name, result, scanning }: {
  src: string; name: string; result?: Analysis; scanning: boolean;
}) {
  const [zoom, setZoom] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [opacity, setOpacity] = useState(.57);
  const [grid, setGrid] = useState(false);
  const [heat, setHeat] = useState(true);
  const drag = useRef<{ x: number; y: number; startX: number; startY: number } | null>(null);
  const tiles = result?.tiles ?? [];
  const low = Math.min(...tiles.map(tile => tile.logit));
  const high = Math.max(...tiles.map(tile => tile.logit));
  const top = result?.top_tile;

  function scale(next: number) {
    const value = Math.max(1, Math.min(3, next));
    setZoom(value);
    if (value === 1) setOffset({ x: 0, y: 0 });
  }

  function pan(x: number, y: number, width: number, height: number) {
    const maxX = width * (zoom - 1) / 2;
    const maxY = height * (zoom - 1) / 2;
    setOffset({ x: Math.max(-maxX, Math.min(maxX, x)), y: Math.max(-maxY, Math.min(maxY, y)) });
  }

  return (
    <>
      <div className="viewer-toolbar" role="group" aria-label="Receipt view controls">
        <button type="button" className="icon-button" aria-label="Zoom out" disabled={zoom === 1} onClick={() => scale(zoom - .25)}><Minus size={17} aria-hidden="true" /></button>
        <button type="button" className="icon-button" aria-label="Zoom in" disabled={zoom === 3} onClick={() => scale(zoom + .25)}><Plus size={17} aria-hidden="true" /></button>
        <button type="button" className="icon-button" aria-label="Reset zoom and pan" disabled={zoom === 1} onClick={() => scale(1)}><RotateCcw size={17} aria-hidden="true" /></button>
        <output aria-label="Zoom level">{Math.round(zoom * 100)}%</output>
        <button type="button" className="icon-button grid-button" aria-label="Show tile grid" aria-pressed={grid}
          disabled={!tiles.length} onClick={() => setGrid(!grid)}><Grid2X2 size={17} aria-hidden="true" /></button>
      </div>
      <div className="stage scan-stage" data-zoomed={zoom > 1} data-lenis-prevent={zoom > 1 ? '' : undefined} tabIndex={0} role="group"
        aria-label="Receipt image. Zoom in, then use arrow keys or drag to pan."
        onPointerDown={(event) => {
          if (zoom === 1 || event.button !== 0) return;
          drag.current = { x: event.clientX, y: event.clientY, startX: offset.x, startY: offset.y };
          event.currentTarget.setPointerCapture(event.pointerId);
        }}
        onPointerMove={(event) => {
          const point = drag.current;
          if (point) pan(point.startX + event.clientX - point.x, point.startY + event.clientY - point.y,
            event.currentTarget.clientWidth, event.currentTarget.clientHeight);
        }}
        onLostPointerCapture={() => { drag.current = null; }}
        onKeyDown={(event) => {
          const delta = { ArrowLeft: [-28, 0], ArrowRight: [28, 0], ArrowUp: [0, -28], ArrowDown: [0, 28] }[event.key];
          if (!delta || zoom === 1) return;
          event.preventDefault();
          pan(offset.x + delta[0], offset.y + delta[1], event.currentTarget.clientWidth, event.currentTarget.clientHeight);
        }}>
        <figure className="sheet receipt-sheet" style={{ transform: 'translate(' + offset.x + 'px,' + offset.y + 'px) scale(' + zoom + ')' }}>
          <img src={src} alt={name} draggable={false} />
          {scanning && <div className="scan-line" aria-hidden="true" />}
          {result && <svg className="tile-layer" viewBox="0 0 1000 1000" preserveAspectRatio="none" aria-hidden="true">
            {heat && <g opacity={opacity}>{tiles.map((tile, index) => (
              <rect key={index} x={tile.x * 1000} y={tile.y * 1000} width={tile.width * 1000} height={tile.height * 1000}
                fill="var(--marker)" opacity={high === low ? .35 : .1 + .85 * (tile.logit - low) / (high - low)} />
            ))}</g>}
            {grid && tiles.map((tile, index) => (
              <rect key={index} x={tile.x * 1000} y={tile.y * 1000} width={tile.width * 1000} height={tile.height * 1000}
                fill="none" stroke="#3a2a19" strokeDasharray="4 3" vectorEffect="non-scaling-stroke" />
            ))}
            {top && <>
              <rect x={top.x * 1000} y={top.y * 1000} width={top.width * 1000} height={top.height * 1000}
                fill="none" stroke="#5b2700" strokeWidth="4" vectorEffect="non-scaling-stroke" />
              <rect x={top.x * 1000} y={top.y * 1000} width={top.width * 1000} height={top.height * 1000}
                fill="none" stroke="#ffca80" strokeWidth="2" vectorEffect="non-scaling-stroke" />
            </>}
          </svg>}
        </figure>
      </div>
      {result && (
        <div className="overlay-controls">
          <span><i aria-hidden="true" /> Highest-scoring tile</span>
          {tiles.length > 0 && <>
            <label><input type="checkbox" checked={heat} onChange={(event) => setHeat(event.currentTarget.checked)} /> Tile heatmap</label>
            <label>Opacity <input type="range" min="0" max="1" step=".01" value={opacity} disabled={!heat}
              onChange={(event) => setOpacity(event.currentTarget.valueAsNumber)} /></label>
            <span className="overlay-hint">Darker tiles have higher relative scores on this receipt.</span>
          </>}
        </div>
      )}
    </>
  );
}
