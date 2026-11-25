'use client';

import Image from 'next/image';
import { FileImage, ScanLine, UploadCloud } from 'lucide-react';
import { useEffect, useId, useRef, useState } from 'react';
import original from '../../public/original.png';
import edited from '../../public/edited.png';
import blurred from '../../public/blurred.png';
import originalThumb from './assets/original-thumb.webp';
import editedThumb from './assets/edited-thumb.webp';
import blurredThumb from './assets/blurred-thumb.webp';
import { Button } from '@/components/ui/button';
import { ReceiptViewer } from '@/components/receipt-viewer';
import { analyze, apiBase, decisionLabels, percent, replay, validateFile, type Analysis, type ReceiptResult, type Sample, type UploadProgress } from '@/lib/receipt-client';

const samples = { edited: { image: edited, thumbnail: editedThumb, name: 'Edited' },
  original: { image: original, thumbnail: originalThumb, name: 'Original' },
  blurred: { image: blurred, thumbnail: blurredThumb, name: 'Blurred' } } as const;
type Selection = { kind: 'sample'; sample: Sample } | { kind: 'file'; file: File; type: string; url: string };
type State = { status: 'idle' | 'validating' } | { status: 'working'; progress: UploadProgress; slow: boolean } |
  { status: 'ready'; result: ReceiptResult } | { status: 'failed'; message: string };

function Findings({ result, saved }: { result: Analysis; saved: boolean }) {
  const probability = percent(result.calibrated_probability);
  return (
    <div className="findings">
      <div className="decision" data-decision={result.decision}>{decisionLabels[result.decision]}</div>
      <div className="probability" role="meter" aria-label="Estimated edit probability" aria-valuemin={0}
        aria-valuemax={100} aria-valuenow={result.calibrated_probability * 100} aria-valuetext={probability}>
        <div className="dial" aria-hidden="true">
          <svg viewBox="0 0 152 82"><path d="M 8 76 A 68 68 0 0 1 144 76" className="dial-base" pathLength={1} />
            <path d="M 8 76 A 68 68 0 0 1 144 76" className="dial-value" pathLength={1}
              style={{ strokeDasharray: result.calibrated_probability + ' 1' }} /></svg>
          <strong>{probability}</strong>
        </div>
        <span>Estimated edit probability</span>
      </div>
      <div className="confidence">
        <label>Binary prediction confidence <strong>{percent(result.confidence)}</strong>
          <meter min={0} max={1} value={result.confidence} /></label>
      </div>
      <p>{result.advice}</p>
      <dl className="timing"><dt>{saved ? 'Recorded analysis time' : 'Analysis time'}</dt>
        <dd>{result.inference_ms.toFixed(2)} ms</dd></dl>
    </div>
  );
}

export default function Demo() {
  const inputId = useId();
  const [selection, setSelection] = useState<Selection | null>(null);
  const [state, setState] = useState<State>({ status: 'idle' });
  const [mode, setMode] = useState<'saved' | 'live'>(apiBase ? 'live' : 'saved');
  const [dragging, setDragging] = useState(false);
  const [version, setVersion] = useState(0);
  const controller = useRef<AbortController | null>(null);
  const uploadUrl = useRef<string | null>(null);
  const validation = useRef(0);
  const slowTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const dragDepth = useRef(0);

  function cancel() {
    controller.current?.abort();
    if (slowTimer.current) clearTimeout(slowTimer.current);
    validation.current++;
  }

  function clearUpload() {
    if (uploadUrl.current) URL.revokeObjectURL(uploadUrl.current);
    uploadUrl.current = null;
  }

  useEffect(() => () => {
    controller.current?.abort();
    if (slowTimer.current) clearTimeout(slowTimer.current);
    if (uploadUrl.current) URL.revokeObjectURL(uploadUrl.current);
    validation.current++;
  }, []);

  async function run(chosen: Selection, currentMode = mode) {
    cancel();
    if (chosen.kind === 'file' && currentMode === 'saved') {
      setState({ status: 'idle' });
      return;
    }
    const task = new AbortController();
    controller.current = task;
    const saved = chosen.kind === 'sample' && currentMode === 'saved';
    setState({ status: 'working', progress: { phase: 'processing' }, slow: false });
    if (!saved) slowTimer.current = setTimeout(() => {
      if (!task.signal.aborted) setState(previous => previous.status === 'working' ? { ...previous, slow: true } : previous);
    }, 5000);

    try {
      const result = saved
        ? await replay(chosen.sample, task.signal)
        : await (async () => {
            const body: Blob = chosen.kind === 'file' ? chosen.file : await fetch(samples[chosen.sample].image.src, { signal: task.signal }).then(response => {
              if (!response.ok) throw new Error('This sample could not be loaded.');
              return response.blob();
            });
            return analyze(body, chosen.kind === 'file' ? chosen.type : 'image/png', task.signal, progress => {
              if (!task.signal.aborted) setState(previous => previous.status === 'working' ? { ...previous, progress } : previous);
            });
          })();
      if (!task.signal.aborted) setState({ status: 'ready', result });
    } catch (error) {
      if (!task.signal.aborted) setState({ status: 'failed',
        message: error instanceof Error ? error.message : 'Analysis could not finish. Try again.' });
    } finally {
      if (controller.current === task && slowTimer.current) clearTimeout(slowTimer.current);
    }
  }

  async function takeFile(file: File) {
    cancel();
    clearUpload();
    setSelection(null);
    const generation = validation.current;
    setState({ status: 'validating' });
    const checked = await validateFile(file);
    if (generation !== validation.current) return;
    if ('outcome' in checked) {
      setState({ status: 'ready', result: checked });
      return;
    }
    const url = URL.createObjectURL(file);
    uploadUrl.current = url;
    const chosen: Selection = { kind: 'file', file, type: checked.type, url };
    setSelection(chosen);
    setVersion(old => old + 1);
    setState({ status: 'idle' });
  }

  function takeSample(sample: Sample, currentMode = mode) {
    clearUpload();
    const chosen: Selection = { kind: 'sample', sample };
    setSelection(chosen);
    setVersion(old => old + 1);
    void run(chosen, currentMode);
  }

  function switchMode(next: 'live' | 'saved') {
    if (next === mode) return;
    setMode(next);
    cancel();
    setState({ status: 'idle' });
    if (selection?.kind === 'sample') takeSample(selection.sample, next);
  }

  const working = state.status === 'working';
  const result = state.status === 'ready' && state.result.outcome === 'analyzed' ? state.result : undefined;
  const preview = selection?.kind === 'sample'
    ? { src: samples[selection.sample].image.src, name: samples[selection.sample].name + ' fictional receipt' }
    : selection?.kind === 'file' ? { src: selection.url, name: 'Selected receipt image' } : null;
  const uploaded = state.status === 'working' && state.progress.phase === 'uploading' ? state.progress.percent : null;

  return (
    <div className="demo-body">
      <div className="demo-tools">
        <div className="mode-switch" role="group" aria-label="Analysis mode">
          <button type="button" aria-pressed={mode === 'saved'} onClick={() => switchMode('saved')}>Saved samples</button>
          {apiBase && <button type="button" aria-pressed={mode === 'live'} onClick={() => switchMode('live')}>Live detector</button>}
        </div>
        <p>{mode === 'saved' ? 'Saved samples replay recorded outputs. Your own file stays in this tab.' : 'Analyzing sends the image to the configured research service.'}</p>
      </div>
      <div className="desk demo-desk">
        <div className="viewer">
          <div className="viewer-bar"><span><FileImage size={16} aria-hidden="true" /> Receipt preview</span>
            {preview && <button type="button" className="text-button" onClick={() => fileInput.current?.click()}>Choose another file</button>}
          </div>
          <input ref={fileInput} id={inputId} type="file" className="sr-only-file" tabIndex={-1}
            aria-label="Choose a receipt image" accept="image/jpeg,image/png,image/webp"
            onChange={(event) => { const file = event.currentTarget.files?.[0]; event.currentTarget.value = ''; if (file) void takeFile(file); }} />
          <div onDragEnter={(event) => { event.preventDefault(); dragDepth.current++; setDragging(true); }}
            onDragOver={(event) => event.preventDefault()} onDragLeave={(event) => { event.preventDefault(); if (--dragDepth.current <= 0) setDragging(false); }}
            onDrop={(event) => { event.preventDefault(); dragDepth.current = 0; setDragging(false);
              const file = event.dataTransfer.files[0]; if (file) void takeFile(file); }}>
            {preview
              ? <ReceiptViewer key={version} {...preview} result={result} scanning={working} />
              : <button type="button" className="dropzone" data-dragging={dragging}
                  onClick={() => fileInput.current?.click()}>
                  <span className="drop-icon"><UploadCloud size={30} strokeWidth={1.4} aria-hidden="true" /></span>
                  <strong>Drop a receipt here</strong><span>or choose a file</span>
                  <small>JPEG, PNG, WebP · up to 8 MiB</small>
                </button>}
          </div>
        </div>
        <aside className="inspector demo-inspector" aria-label="Receipt findings">
          <div>
            <h3>Try a fictional receipt</h3>
            <div className="specimens demo-specimens" role="group" aria-label="Choose a fictional receipt">
              {(Object.keys(samples) as Sample[]).map(id => <button type="button" className="specimen" key={id}
                aria-pressed={selection?.kind === 'sample' && selection.sample === id} onClick={() => takeSample(id)}>
                <Image src={samples[id].thumbnail} alt="" width={30} height={43} />{samples[id].name}
              </button>)}
            </div>
          </div>
          <div className="result-slot" aria-live="polite">
            {state.status === 'validating' && <p>Checking the image…</p>}
            {working && <div className="processing-panel" role="status">
              <ScanLine size={23} aria-hidden="true" />
              <p>{state.progress.phase === 'uploading' ? 'Uploading receipt…' : 'Inspecting receipt…'}</p>
              {state.progress.phase === 'uploading' && <progress max={100} value={uploaded ?? undefined} aria-label="Upload progress" />}
              {state.slow && <p>The research server may be waking up. Keep this tab open or switch to saved samples.</p>}
              <Button type="button" variant="outline" onClick={() => { cancel(); setState({ status: 'idle' }); }}>Cancel</Button>
            </div>}
            {state.status === 'failed' && <div className="outcome-message">
              <h3>Analysis unavailable</h3><p>{state.message}</p>
              {selection && <Button type="button" onClick={() => run(selection)}>Retry analysis</Button>}
            </div>}
            {state.status === 'ready' && state.result.outcome === 'unsupported_input' &&
              <div className="outcome-message" data-state="unsupported"><h3>Unsupported input</h3><p>{state.result.advice}</p></div>}
            {result && <Findings result={result} saved={mode === 'saved'} />}
            {state.status === 'idle' && (selection
              ? mode === 'saved' ? <p>Preview only. Choose a fictional sample to see saved analysis.</p>
                : <Button type="button" onClick={() => run(selection)}><ScanLine size={17} aria-hidden="true" /> Analyze receipt</Button>
              : <p>Choose a sample or add an image to start.</p>)}
          </div>
        </aside>
      </div>
    </div>
  );
}
