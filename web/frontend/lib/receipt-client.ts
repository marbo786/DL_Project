export type Bounds = { x: number; y: number; width: number; height: number };
export type Tile = Bounds & { logit: number };
export type Decision = 'prioritize_review' | 'normal_queue' | 'inconclusive';
export const decisionLabels: Record<Decision, string> = {
  prioritize_review: 'Prioritize review',
  normal_queue: 'Normal queue',
  inconclusive: 'Inconclusive',
};
export const percent = (value: number) => (value * 100).toFixed(2) + '%';
export type Analysis = {
  outcome: 'analyzed';
  analysis_id: string;
  model_version: string;
  policy_version: string;
  decision: Decision;
  calibrated_probability: number;
  confidence: number;
  advice: string;
  inference_ms: number;
  image_width: number;
  image_height: number;
  top_tile: Bounds;
  tiles: Tile[] | null;
};

export type Unsupported = {
  outcome: 'unsupported_input';
  reason: 'unreadable' | 'unsupported_format' | 'animated' | 'too_large' | 'empty';
  advice: string;
};
export type ReceiptResult = Analysis | Unsupported;
export type Sample = 'original' | 'edited' | 'blurred';
export type UploadProgress = { phase: 'uploading'; percent: number | null } | { phase: 'processing' };
export const apiBase = (process.env.NEXT_PUBLIC_FORGERYLENS_API_URL ?? '').replace(/\/+$/, '');
const maxBytes = 8 * 1024 * 1024;
const maxPixels = 16_000_000;
const mediaTypes = new Map([['png', 'image/png'], ['jpeg', 'image/jpeg'], ['jpg', 'image/jpeg'], ['webp', 'image/webp']]);

const invalid = (reason: Unsupported['reason'], advice: string): Unsupported => ({ outcome: 'unsupported_input', reason, advice });
const object = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object';
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const fraction = (value: unknown): value is number => finite(value) && value >= 0 && value <= 1;
const bounds = (value: unknown): value is Bounds => object(value) &&
  fraction(value.x) && fraction(value.y) && fraction(value.width) && fraction(value.height) &&
  value.width > 0 && value.height > 0 && value.x + value.width <= 1.000001 && value.y + value.height <= 1.000001;
const decisions = ['prioritize_review', 'normal_queue', 'inconclusive'];
const reasons = ['unreadable', 'unsupported_format', 'animated', 'too_large', 'empty'];

function parseResult(value: unknown): ReceiptResult {
  if (object(value) && typeof value.advice === 'string') {
    if (value.outcome === 'unsupported_input' && reasons.includes(String(value.reason))) return value as Unsupported;
    if (value.outcome === 'analyzed' && decisions.includes(String(value.decision)) &&
        typeof value.analysis_id === 'string' && typeof value.model_version === 'string' &&
        typeof value.policy_version === 'string' && fraction(value.calibrated_probability) && fraction(value.confidence) &&
        finite(value.inference_ms) && value.inference_ms >= 0 && finite(value.image_width) && value.image_width > 0 &&
        finite(value.image_height) && value.image_height > 0 && bounds(value.top_tile) &&
        (value.tiles === null || (Array.isArray(value.tiles) && value.tiles.every(
          (tile: unknown) => object(tile) && finite(tile.logit) && bounds(tile))))) {
      return value as Analysis;
    }
  }
  throw new Error('The receipt service returned an unreadable result.');
}

export async function validateFile(file: File): Promise<Unsupported | { type: string }> {
  const type = file.type || mediaTypes.get(file.name.split('.').at(-1)?.toLowerCase() ?? '');
  if (!file.size) return invalid('empty', 'Choose a nonempty image.');
  if (!type || ![...mediaTypes.values()].includes(type)) return invalid('unsupported_format', 'Choose a JPEG, PNG, or WebP image.');
  if (file.size > maxBytes) return invalid('too_large', 'Choose an image no larger than 8 MiB.');
  if ('ImageDecoder' in window) {
    const Decoder = window.ImageDecoder as new (options: { data: ReadableStream<Uint8Array>; type: string }) => {
      tracks: { ready: Promise<void>; selectedTrack: { frameCount: number } | null }; close: () => void;
    };
    let decoder;
    try {
      decoder = new Decoder({ data: file.stream(), type });
      await decoder.tracks.ready;
      if ((decoder.tracks.selectedTrack?.frameCount ?? 0) > 1) return invalid('animated', 'Choose a single-frame image.');
    } catch {
      // A still-image decoder can handle browsers with partial ImageDecoder support.
    } finally {
      decoder?.close();
    }
  }
  try {
    const image = await createImageBitmap(file, { imageOrientation: 'from-image' });
    const pixels = image.width * image.height;
    image.close();
    if (pixels > maxPixels) return invalid('too_large', 'Choose an image of at most 16 megapixels.');
  } catch {
    return invalid('unreadable', 'This image could not be opened. Try another copy.');
  }
  return { type };
}

export async function replay(sample: Sample, signal: AbortSignal): Promise<Analysis> {
  const response = await fetch('/data/sample-results.json', { signal });
  if (!response.ok) throw new Error('Saved specimen results are unavailable. Try again.');
  const data: unknown = await response.json();
  const entry = object(data) ? data[sample] : null;
  const result = parseResult(object(entry) ? entry.result : null);
  if (result.outcome !== 'analyzed') throw new Error('No analysis was saved for this specimen.');
  return result;
}

export function analyze(file: Blob, type: string, signal: AbortSignal, progress: (value: UploadProgress) => void) {
  return new Promise<ReceiptResult>((resolve, reject) => {
    const request = new XMLHttpRequest();
    const stop = () => request.abort();
    request.open('POST', apiBase + '/api/v1/receipt/analyze');
    request.timeout = 75_000;
    request.responseType = 'json';
    request.setRequestHeader('Content-Type', type);
    request.upload.onprogress = (event) => progress({
      phase: 'uploading', percent: event.lengthComputable ? Math.round(100 * event.loaded / event.total) : null,
    });
    request.upload.onload = () => progress({ phase: 'processing' });
    request.onload = () => {
      try {
        if ([200, 400, 413, 415].includes(request.status)) return resolve(parseResult(request.response));
        if (request.status === 429) throw new Error('The detector is busy. Try again shortly.');
        if (request.status === 503) throw new Error('The detector is unavailable. Retry or switch to saved samples.');
        if (request.status === 408) throw new Error('The upload was interrupted. Try again.');
        throw new Error('The detector could not complete this request.');
      } catch (error) { reject(error); }
    };
    request.onerror = () => reject(new Error('The service could not be reached. Check the connection or try saved samples.'));
    request.ontimeout = () => reject(new Error('The service took too long. Retry or switch to saved samples.'));
    request.onabort = () => reject(new DOMException('Cancelled', 'AbortError'));
    request.onloadend = () => signal.removeEventListener('abort', stop);
    if (signal.aborted) return reject(new DOMException('Cancelled', 'AbortError'));
    signal.addEventListener('abort', stop, { once: true });
    progress({ phase: 'uploading', percent: null });
    request.send(file);
  });
}
