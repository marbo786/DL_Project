const main = document.querySelector("main");
const route = location.pathname.replace(/\/$/, "") || "/";
document.querySelectorAll(".links a").forEach(link => {
  if (link.pathname === route) link.setAttribute("aria-current", "page");
});

const assets = "/assets/";
const samples = [
  { name: "Original specimen", file: "original.png" },
  { name: "Edited specimen", file: "edited.png", region: { x: .09, y: .68, width: .82, height: .09 } },
  { name: "Low-quality specimen", file: "blurred.png" },
];

async function loadSpecimen() {
  try {
    const response = await fetch(assets + "specimen.json");
    if (!response.ok) return;
    const sample = await response.json();
    if (sample.edit_region) samples[1].region = sample.edit_region;
  } catch {
    // The original illustrative sample still works without metadata.
  }
}
const specimenReady = loadSpecimen();

const layouts = {
  "/methodology": ["Methodology", [
    ["Source groups", "Each original document and its variants belong to one partition. The external receipt holdout changes document corpus within the same editing generator."],
    ["Model comparison", "Image-only receipt models ranked external examples near chance. The local Reference Studio compares two images using classical registered SSIM."],
    ["Limits of the evidence", "Reference experiments applied simulated acquisition changes. Independent physical-capture accuracy is still unmeasured. A visible difference can have a legitimate cause."],
  ]],
  "/about": ["About ForgeryLens", [
    ["AI341 research prototype", "ForgeryLens is a document image inspection project for the Deep Neural Networks course at GIKI."],
    ["Use and limitations", "Use any findings to guide manual review. The system evaluates image patterns and cannot certify authenticity or recover original values."],
  ]],
  "/privacy": ["Document handling", [
    ["Local preview", "The Studio displays selected images in the browser. The image is sent to the configured analysis service when you choose Analyze."],
    ["Analysis service", "The service processes images in memory and keeps no upload history. Hosting providers may collect ordinary request metadata. Use demo specimens for public presentations."],
  ]],
};

function homepage() {
  main.innerHTML = `
    <section class="hero">
      <div>
        <h1>Inspect document changes</h1>
        <p>ForgeryLens is a research prototype. The local Reference Studio compares visible differences between two images. The historical single-image demo uses fictional receipts.</p>
        <div class="actions"><a class="button primary" href="/studio">Open Forensic Studio</a><a class="button" href="/research">Explore the Research</a></div>
      </div>
      <div class="stage hero-preview">
        <div class="sample-wrap"><img src="${assets}edited.png" alt="Fictional receipt specimen"><span class="region" id="hero-region" aria-hidden="true"></span></div>
        <label><input type="checkbox" id="hero-toggle" checked> Show edit annotation</label>
        <p class="caption">Fictional specimen · manual annotation</p>
      </div>
    </section>
    <section class="info-grid" aria-label="Inspection features">
      <article><h2>Inspect the document</h2><p>Zoom in and pan across receipts or supported form images.</p></article>
      <article><h2>Review visible differences</h2><p>In local research mode, align a candidate to a trusted reference and inspect structural differences in the overlap.</p></article>
      <article><h2>Account for uncertainty</h2><p>Use inconclusive results to flag documents for further review.</p></article>
    </section>`;
  specimenReady.then(() => {
    const region = document.querySelector("#hero-region");
    const box = samples[1].region;
    for (const [style, key] of [["left", "x"], ["top", "y"], ["width", "width"], ["height", "height"]]) {
      region.style[style] = box[key] * 100 + "%";
    }
  });
  document.querySelector("#hero-toggle").addEventListener("change", event => {
    document.querySelector("#hero-region").hidden = !event.target.checked;
  });
}

async function api(path, options) {
  const response = await fetch("/api/v1/" + path, { ...options, signal: AbortSignal.timeout(75000) });
  let data;
  try { data = await response.json(); } catch { throw new Error("The analysis service is unavailable. Try again shortly."); }
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "The service could not complete the request.");
  return data;
}

function studio() {
  main.innerHTML = `
    <div class="mast"><div><h1>Forensic Studio</h1><p>Historical single-image demonstration on fictional receipts.</p></div>
    <span class="status" id="model-status" role="status">Checking model</span></div>
    <div class="desk">
      <section class="viewer" aria-label="Document viewer">
        <div class="toolbar"><div class="group">
          <button class="button" id="zoom-out" aria-label="Zoom out">−</button><span class="readout" id="zoom">Fit</span>
          <button class="button" id="zoom-in" aria-label="Zoom in">+</button><button class="button" id="fit">Fit</button><button class="button" id="actual">Actual size</button>
        </div><span class="readout" id="coordinates"></span></div>
        <div class="stage viewport" id="viewport" tabindex="0" aria-label="Document stage. Use arrow keys to pan. Drop an image to preview.">
          <div class="sheet" id="sheet" hidden><img id="image" alt=""><canvas id="heatmap" hidden aria-hidden="true"></canvas><div class="regions" id="regions" aria-hidden="true"></div></div>
        </div>
        <div class="notice" id="notice" role="status"></div>
      </section>
      <aside class="evidence" aria-label="Analysis and evidence">
        <section><h2>Document</h2><p>Drop a JPEG, PNG or WebP image onto the viewer. Up to 8 MiB and 16 MP.</p>
          <div class="actions"><button class="button" id="choose">Choose image</button><button class="button primary" id="analyze" disabled>Analyze</button></div>
          <input type="file" id="file" accept="image/jpeg,image/png,image/webp" hidden>
        </section>
        <section><h2>Evidence</h2><div id="finding" aria-live="polite"><p>Checking availability.</p></div>
          <div class="controls">
            <label><input type="checkbox" id="region-toggle" checked> Show regions</label>
            <label><input type="checkbox" id="heat-toggle" checked disabled> Show heatmap</label>
            <label>Heatmap opacity <input type="range" id="opacity" min="0" max="100" value="55" disabled></label>
          </div>
        </section>
        <section><h2>Demo specimens</h2><div class="specimens">${samples.map((sample, index) =>
          `<button class="button" data-sample="${index}" aria-pressed="false">${sample.name}</button>`).join("")}</div></section>
      </aside>
    </div>`;
  const get = id => document.getElementById(id);
  const viewport = get("viewport"), sheet = get("sheet"), image = get("image");
  const regions = get("regions"), canvas = get("heatmap"), finding = get("finding");
  const analyze = get("analyze"), fileInput = get("file");
  const notice = get("notice"), coordinateLabel = get("coordinates");
  const regionToggle = get("region-toggle"), heatToggle = get("heat-toggle"), opacity = get("opacity");
  let documentImage = null, uploaded = null, selected = null, objectURL = null, revision = 0;
  let ready = false, busy = false, scale = 1, fitMode = true, dx = 0, dy = 0, drag = null;

  function message(text, error = false) {
    notice.textContent = text;
    notice.classList.toggle("error", error);
  }

  function updateButton() {
    analyze.disabled = !ready || !documentImage || busy;
    analyze.textContent = busy ? "Analyzing…" : "Analyze";
  }

  function transform() {
    if (!documentImage) return;
    const { naturalWidth, naturalHeight } = documentImage;
    if (fitMode) scale = Math.min((viewport.clientWidth - 44) / naturalWidth, (viewport.clientHeight - 42) / naturalHeight, 1);
    sheet.style.width = naturalWidth + "px";
    sheet.style.height = naturalHeight + "px";
    sheet.style.transform = `translate(calc(-50% + ${dx}px), calc(-50% + ${dy}px)) scale(${scale})`;
    get("zoom").textContent = Math.round(scale * 100) + "%";
  }

  function reset(toFit) {
    fitMode = toFit;
    scale = 1;
    dx = dy = 0;
    transform();
  }

  function zoom(factor) {
    if (!documentImage) return;
    fitMode = false;
    scale = Math.min(4, Math.max(Math.min(.06, scale), scale * factor));
    transform();
  }

  function draw(items, heatmap) {
    regions.replaceChildren();
    for (const item of items) {
      const box = document.createElement("span");
      box.className = "region";
      for (const [style, key] of [["left", "x"], ["top", "y"], ["width", "width"], ["height", "height"]]) {
        box.style[style] = item[key] * 100 + "%";
      }
      regions.append(box);
    }
    regionToggle.disabled = items.length === 0;
    regions.hidden = !regionToggle.checked;
    heatToggle.disabled = opacity.disabled = !heatmap;
    canvas.hidden = !heatmap || !heatToggle.checked;
    canvas.style.opacity = Number(opacity.value) / 100;
    if (!heatmap) return;
    canvas.width = heatmap.width;
    canvas.height = heatmap.height;
    const context = canvas.getContext("2d");
    const pixels = context.createImageData(canvas.width, canvas.height);
    heatmap.values.forEach((value, i) => {
      pixels.data.set([255, Math.round(173 - value * 99), 43, Math.round(value * 255)], i * 4);
    });
    context.putImageData(pixels, 0, 0);
  }

  function showText(text) {
    const paragraph = document.createElement("p");
    paragraph.textContent = text;
    finding.replaceChildren(paragraph);
  }

  function showDefault() {
    if (selected?.region) showText("The outline marks the manually edited field in this fictional specimen.");
    else if (!ready) showText("Live analysis requires a trained model.");
    else showText("Choose Analyze to check this image.");
  }

  async function open({ url, name, file = null, sample = null }) {
    const ticket = ++revision;
    if (objectURL) URL.revokeObjectURL(objectURL);
    objectURL = file ? url : null;
    documentImage = uploaded = selected = null;
    sheet.hidden = true;
    coordinateLabel.textContent = "";
    draw([], null);
    updateButton();
    message("Opening image…");
    showDefault();
    const preview = new Image();
    preview.src = url;
    try { await preview.decode(); } catch {
      if (ticket === revision) message("This image could not be opened. Choose a readable JPEG, PNG or WebP file.", true);
      return;
    }
    if (ticket !== revision) return;
    if (preview.naturalWidth * preview.naturalHeight > 16000000) {
      message("This image exceeds 16 megapixels.", true);
      return;
    }
    documentImage = preview;
    uploaded = file;
    selected = sample;
    image.src = url;
    image.alt = sample ? name : "Selected document";
    sheet.hidden = false;
    reset(true);
    draw(sample?.region ? [sample.region] : [], null);
    message(`${name} · ${preview.naturalWidth} × ${preview.naturalHeight} px`);
    showDefault();
    updateButton();
  }

  async function selectSample(index) {
    if (index === 1) await specimenReady;
    const sample = samples[index];
    document.querySelectorAll("[data-sample]").forEach(button => button.setAttribute("aria-pressed", Number(button.dataset.sample) === index));
    void open({ url: assets + sample.file, name: sample.name, sample });
  }

  function selectFile(file) {
    if (!file) return;
    if (!file.size || file.size > 8 * 1024 * 1024) return message("Choose a nonempty image no larger than 8 MiB.", true);
    if (file.type && !["image/jpeg", "image/png", "image/webp"].includes(file.type)) return message("Choose a JPEG, PNG or WebP image.", true);
    document.querySelectorAll("[data-sample]").forEach(button => button.setAttribute("aria-pressed", "false"));
    void open({ url: URL.createObjectURL(file), name: file.name, file });
  }

  function showResult(result) {
    const title = document.createElement("h3");
    title.textContent = { no_strong_evidence: "No strong evidence", suspicious: "Suspicious", inconclusive: "Inconclusive" }[result.decision];
    const rows = document.createElement("dl");
    const percent = value => value == null ? "Unavailable" : (value * 100).toFixed(1) + "%";
    for (const [key, value] of [
      ["Model", result.model_version], ["Inference", result.inference_ms + " ms"],
      ["Manipulation probability", percent(result.calibrated_probability)], ["Uncertainty", percent(result.uncertainty)],
      ["Candidate regions", result.regions.length],
    ]) {
      const term = document.createElement("dt"), detail = document.createElement("dd");
      term.textContent = key;
      detail.textContent = value;
      rows.append(term, detail);
    }
    finding.replaceChildren(title, rows);
    for (const warning of [...result.warnings, "Use this evidence to guide manual review."]) {
      const item = document.createElement("p");
      item.textContent = warning;
      finding.append(item);
    }
  }

  analyze.addEventListener("click", async () => {
    if (!ready || busy || !documentImage) return;
    const ticket = revision;
    busy = true;
    draw([], null);
    showText("Analyzing document…");
    updateButton();
    try {
      const body = uploaded ?? await (await fetch(documentImage.src)).blob();
      const result = await api("analyze", { method: "POST", body, headers: { "Content-Type": body.type || "application/octet-stream" } });
      if (revision !== ticket) return;
      showResult(result);
      if (result.image_width === documentImage.naturalWidth && result.image_height === documentImage.naturalHeight) {
        draw(result.regions, result.heatmap);
      } else {
        message("Overlay unavailable because the server and browser report different image dimensions.", true);
      }
    } catch (error) {
      if (ticket === revision) {
        showDefault();
        draw(selected?.region ? [selected.region] : [], null);
        message(error.message, true);
      }
    } finally {
      busy = false;
      updateButton();
    }
  });

  get("choose").addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => { selectFile(fileInput.files[0]); fileInput.value = ""; });
  document.querySelectorAll("[data-sample]").forEach(button => button.addEventListener("click", () => selectSample(Number(button.dataset.sample))));
  get("zoom-in").addEventListener("click", () => zoom(1.25));
  get("zoom-out").addEventListener("click", () => zoom(.8));
  get("fit").addEventListener("click", () => reset(true));
  get("actual").addEventListener("click", () => reset(false));
  regionToggle.addEventListener("change", () => { regions.hidden = !regionToggle.checked; });
  heatToggle.addEventListener("change", () => { canvas.hidden = !heatToggle.checked; });
  opacity.addEventListener("input", () => { canvas.style.opacity = Number(opacity.value) / 100; });
  viewport.addEventListener("dragover", event => event.preventDefault());
  viewport.addEventListener("drop", event => { event.preventDefault(); selectFile(event.dataTransfer.files[0]); });
  viewport.addEventListener("pointerdown", event => {
    if (!documentImage || event.button !== 0 || drag) return;
    drag = { id: event.pointerId, x: event.clientX - dx, y: event.clientY - dy };
    viewport.setPointerCapture(event.pointerId);
  });
  viewport.addEventListener("pointermove", event => {
    if (!documentImage) return;
    if (drag?.id === event.pointerId) {
      dx = event.clientX - drag.x;
      dy = event.clientY - drag.y;
      transform();
    }
    const rect = sheet.getBoundingClientRect();
    const x = Math.floor((event.clientX - rect.left) / scale);
    const y = Math.floor((event.clientY - rect.top) / scale);
    coordinateLabel.textContent = x >= 0 && y >= 0 && x < documentImage.naturalWidth && y < documentImage.naturalHeight ? `x ${x} / y ${y}` : "";
  });
  viewport.addEventListener("lostpointercapture", event => { if (drag?.id === event.pointerId) drag = null; });
  viewport.addEventListener("pointerleave", () => { coordinateLabel.textContent = ""; });
  viewport.addEventListener("keydown", event => {
    const offset = { ArrowLeft: [-24, 0], ArrowRight: [24, 0], ArrowUp: [0, -24], ArrowDown: [0, 24] }[event.key];
    if (!offset || !documentImage) return;
    event.preventDefault();
    dx += offset[0];
    dy += offset[1];
    transform();
  });
  new ResizeObserver(transform).observe(viewport);
  selectSample(0);
  api("model-info").then(info => {
    ready = info.available;
    get("model-status").textContent = ready ? (info.scope === "synthetic" ? "Synthetic demo model ready" : "Model ready") : "Model unavailable";
    get("model-status").classList.toggle("ready", ready);
    updateButton();
    showDefault();
  }).catch(() => {
    get("model-status").textContent = "Service unavailable";
    showDefault();
  });
}


const analysisLabels = {
  local: "Validation-selected local model",
  baseline: "Whole-image baseline",
  fp16_storage: "FP16 weight storage",
  distilled_student: "Distilled student",
  grayscale_only: "Grayscale only",
  average_pooling: "Average pooling",
  no_global_context: "Local evidence only",
  classification_only: "Classification loss only",
};
const formatMetric = (value, digits = 3) => typeof value === "number" ? value.toFixed(digits) : "—";

function resultTable(headers, rows) {
  const table = document.createElement("table");
  const head = table.createTHead().insertRow();
  for (const title of headers) {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = title;
    head.append(cell);
  }
  const body = table.createTBody();
  for (const values of rows) {
    const row = body.insertRow();
    for (const value of values) row.insertCell().textContent = value;
  }
  const wrapper = document.createElement("div");
  wrapper.className = "metric-table";
  wrapper.append(table);
  return wrapper;
}


function metricPlot(label, xLabel, yLabel, data, reference = false, maxX = 1) {
  const ns = "http://www.w3.org/2000/svg";
  const chart = document.createElementNS(ns, "svg");
  chart.setAttribute("viewBox", "0 0 360 230");
  chart.setAttribute("role", "img");
  chart.setAttribute("aria-label", label);
  function add(tag, attrs, content) {
    const item = document.createElementNS(ns, tag);
    for (const [name, value] of Object.entries(attrs)) item.setAttribute(name, value);
    if (content) item.textContent = content;
    chart.append(item);
  }
  const x = value => 54 + Math.min(maxX, Math.max(0, value)) / maxX * 276;
  const y = value => 174 - Math.min(1, Math.max(0, value)) * 140;
  for (const step of [0, .25, .5, .75, 1]) {
    add("line", { x1: 54, x2: 330, y1: y(step), y2: y(step), class: "grid" });
    add("text", { x: 45, y: y(step) + 4, "text-anchor": "end" }, step.toFixed(2));
    add("text", { x: x(step * maxX), y: 191, "text-anchor": "middle" }, (step * maxX).toFixed(2));
  }
  if (reference) add("line", { x1: x(0), x2: x(1), y1: y(0), y2: y(1), class: "reference" });
  for (const point of data) {
    if (typeof point.x !== "number" || typeof point.y !== "number") continue;
    add("circle", { cx: x(point.x), cy: y(point.y), r: 4.5, class: point.dominated ? "dominated" : "value" });
  }
  add("text", { x: 192, y: 215, "text-anchor": "middle" }, xLabel);
  add("text", { x: 54, y: 20 }, yLabel);
  const fig = document.createElement("figure");
  fig.className = "metric-plot";
  const title = document.createElement("figcaption");
  title.textContent = label;
  fig.append(title, chart);
  return fig;
}

function renderMetrics(section, key, report) {
  if (key === "summary" && report.models) {
    section.append(resultTable(["Model", "Clean F1", "Shifted F1", "Parameters", "CPU median"],
      Object.entries(report.models).map(([name, row]) => [
        analysisLabels[name] || name, formatMetric(row.test?.macro_f1), formatMetric(row.shifted_test?.macro_f1),
        row.params.toLocaleString(), formatMetric(row.timing?.median_ms, 2) + " ms",
      ])));
    const plotRows = Object.values(report.models).map(row => ({
      x: row.timing?.median_ms, y: row.validation?.macro_f1, dominated: row.pareto_dominated,
    })).filter(row => typeof row.x === "number" && typeof row.y === "number");
    if (plotRows.length) section.append(metricPlot("Validation efficiency comparison", "Median CPU milliseconds", "Macro F1",
      plotRows, false, Math.max(.1, ...plotRows.map(row => row.x)) * 1.1));

  }
  if (key === "ablations" && report.validation_aggregate) {
    const display = metric => metric ? formatMetric(metric.mean) + " ± " + formatMetric(metric.std) : "—";
    section.append(resultTable(["Variant", "Validation F1", "ROC-AUC", "Mask IoU"],
      Object.entries(report.validation_aggregate).map(([name, data]) => [
        name === "local" ? "Full local model" : analysisLabels[name] || name, display(data.macro_f1), display(data.roc_auc), display(data.mean_mask_iou),
      ])));
  }
  if (key === "robustness" && report.stress) {
    section.append(resultTable(["Condition", "Macro F1", "Manipulated recall", "Mask IoU"],
      [["Clean", report.clean], ...Object.entries(report.stress), ["Held-out edit operator", report.shifted]].map(([name, row]) => [
        name.replaceAll("_", " "), formatMetric(row.macro_f1), formatMetric(row.manipulated_recall), formatMetric(row.mean_mask_iou),
      ])));
    if (report.shifted_failure_cases?.length) {
      section.append(resultTable(["Shifted index", "Expected", "Predicted", "Mask IoU"],
        report.shifted_failure_cases.map(row => [row.shifted_test_index, row.expected ? "Edited" : "Original",
          row.predicted ? "Edited" : "Original", formatMetric(row.mask_iou)])));
      if (report.scope === "synthetic") {
        const gallery = document.createElement("div");
        gallery.className = "failure-gallery";
        for (const row of report.shifted_failure_cases) {
          const item = document.createElement("figure");
          const image = document.createElement("img");
          image.loading = "lazy";
          image.src = assets + "failure-" + Number(row.shifted_test_index) + ".png";
          image.alt = "Fictional shifted receipt, model evidence in amber and edit extent in cyan";
          const caption = document.createElement("figcaption");
          caption.textContent = "Shifted specimen " + row.shifted_test_index;
          item.append(image, caption);
          gallery.append(item);
        }
        section.append(gallery);
      }

    }
  }
  if (key === "calibration" && report.test_before && report.test_after) {
    section.append(resultTable(["Condition", "ECE", "Coverage", "Accepted error"],
      [["Clean, before", report.test_before], ["Clean, after", report.test_after], ["Shifted, after", report.shifted_after]]
        .map(([name, row]) => [name, formatMetric(row.ece), formatMetric(row.selective?.coverage), formatMetric(row.selective?.error_rate)])));
    const plots = document.createElement("div");
    plots.className = "plot-row";
    if (report.test_after.reliability?.length) plots.append(metricPlot("Clean-test reliability", "Mean predicted probability", "Observed edit fraction",
      report.test_after.reliability.map(row => ({ x: row.mean_probability, y: row.observed_fraction })), true));
    if (report.coverage_curve?.length) plots.append(metricPlot("Validation risk and coverage", "Accepted fraction", "Error among accepted",
      report.coverage_curve.filter(row => row.error_rate !== null).map(row => ({ x: row.coverage, y: row.error_rate }))));
    section.append(plots);

  }
}

function research() {
  main.innerHTML = `<section class="copy"><h1>Research explorer</h1>
    <p>The artifacts below describe the historical synthetic release. The final report covers the external receipt experiments, simulated reference tests, and the remaining independent evaluation.</p>
    <p><a href="https://github.com/marbo786/Forgery_Lens/blob/main/docs/history/FINAL_RESEARCH_REPORT.md">Read the historical research report</a></p>
    <div id="reports"></div><p><a href="/methodology">Read the methodology</a></p></section>`;
  for (const [key, title] of [
    ["summary", "Dataset, baseline and efficiency"], ["ablations", "Controlled ablations"],
    ["robustness", "Robustness and failure cases"], ["calibration", "Calibration and rejection"],
  ]) {
    const section = document.createElement("section");
    section.className = "report-row";
    const heading = document.createElement("h2"), status = document.createElement("p");
    heading.textContent = title;
    status.textContent = "Checking for results…";
    section.append(heading, status);
    document.querySelector("#reports").append(section);
    api("experiments/" + key).then(result => {
      if (result.status !== "available") { status.textContent = "Results pending."; return; }
      status.textContent = (result.artifact.scope === "synthetic" ? "Fictional receipt benchmark · " : "Run ") + result.artifact.run_id;
      renderMetrics(section, key, result.artifact);
      const detail = document.createElement("details"), label = document.createElement("summary");
      label.textContent = "View run artifact";
      const content = document.createElement("pre");
      content.textContent = JSON.stringify(result.artifact, null, 2);
      detail.append(label, content);
      section.append(detail);
    }).catch(() => { status.textContent = "Research service unavailable."; });
  }
}

if (route === "/studio" || route === "/studio/single") studio();
else if (route === "/research") research();
else if (layouts[route]) {
  const [title, sections] = layouts[route];
  main.innerHTML = `<section class="copy"><h1>${title}</h1>${sections.map(([heading, content]) => `<h2>${heading}</h2><p>${content}</p>`).join("")}</section>`;
} else homepage();
