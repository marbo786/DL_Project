const get = (id) => document.getElementById(id);
const form = get("comparison-form");
const submit = form.querySelector("[type=submit]");
const status = get("comparison-status");
const state = {files: {}, urls: {}, generation: 0, accepted: false, zoom: 1, x: 0, y: 0};
const examples = {
  unchanged: ["north-quay", "north-quay", "Unchanged fictional receipt."],
  altered: ["north-quay", "changed-total", "Fictional receipt with a deliberately changed total."],
  misaligned: ["north-quay", "misaligned", "Fictional receipt with a severe crop and rotation."],
  different: ["different-reference", "north-quay", "These fictional documents have different identities by design. A high difference score cannot establish reference identity or confirm a valid pairing."],
};
const viewports = [get("reference-view"), get("candidate-view")];
const reference = viewports[0].querySelector("img");
const candidate = viewports[1].querySelector(".candidate");
const overlayImages = {
  heat: viewports[1].querySelector(".heat"),
  regions: viewports[1].querySelector(".regions"),
  overlap: viewports[1].querySelector(".overlap"),
};
const before = viewports[1].querySelector(".before");

function paintCamera() {
  const max = (state.zoom - 1) * 50;
  state.x = Math.min(max, Math.max(-max, state.x));
  state.y = Math.min(max, Math.max(-max, state.y));
  for (const viewport of viewports) {
    viewport.style.setProperty("--zoom", state.zoom);
    viewport.style.setProperty("--x", state.x + "%");
    viewport.style.setProperty("--y", state.y + "%");
    viewport.classList.toggle("zoomed", state.zoom > 1);
  }
  get("zoom-level").value = Math.round(state.zoom * 100) + "%";
  get("zoom-in").disabled = !Object.keys(state.files).length || state.zoom >= 4;
  get("zoom-out").disabled = state.zoom <= 1;
  get("reset-view").disabled = state.zoom <= 1;
}

function setView(swipe) {
  const active = swipe && state.accepted;
  get("viewers").classList.toggle("swipe", active);
  get("side-by-side").setAttribute("aria-pressed", String(!active));
  get("swipe-view").setAttribute("aria-pressed", String(active));
  before.hidden = !active;
  viewports[1].querySelector(".split-line").hidden = !active;
  get("swipe-row").hidden = !active;
  get("candidate-title").textContent = active ? "Reference / aligned candidate"
    : state.accepted ? "Aligned candidate" : "Candidate";
}

function showInputs() {
  for (const [i, name] of ["reference", "candidate"].entries()) {
    const url = state.urls[name];
    const viewport = viewports[i];
    const image = i ? candidate : reference;
    if (url) image.src = url;
    else image.removeAttribute("src");
    viewport.querySelector(".layer").hidden = !url;
    viewport.querySelector(".empty").hidden = Boolean(url);
    form.querySelector('[data-slot="' + name + '"] .filename').textContent =
      state.files[name]?.name ?? "Drop an image or choose a file";
  }
  submit.disabled = !state.files.reference || !state.files.candidate;
  get("clear-pair").disabled = !Object.keys(state.files).length;
  paintCamera();
}

function resetResult() {
  state.generation += 1;
  state.accepted = false;
  get("comparison-decision").textContent = "Ready for comparison";
  get("comparison-decision").removeAttribute("data-state");
  get("comparison-detail").textContent = "Check the reference and candidate, then compare.";
  get("diagnostics").hidden = true;
  get("visual-evidence").hidden = true;
  get("swipe-view").disabled = true;
  get("overlays").disabled = true;
  for (const image of Object.values(overlayImages)) {
    image.hidden = true;
    image.removeAttribute("src");
  }
  before.querySelector("img").removeAttribute("src");
  setView(false);
  showInputs();
}

function assignFile(name, file) {
  if (state.urls[name]) URL.revokeObjectURL(state.urls[name]);
  state.files[name] = file;
  state.urls[name] = URL.createObjectURL(file);
}

function describeExample(key) {
  const note = get("example-note");
  note.textContent = key ? examples[key][2] : "";
  note.hidden = !key;
  for (const button of form.querySelectorAll("[data-example]")) {
    button.setAttribute("aria-pressed", String(button.dataset.example === key));
  }
}

function updateFile(name, file) {
  if (!file) return;
  if (!file.size || file.size > 8 * 1024 * 1024) {
    status.textContent = "Choose a nonempty image no larger than 8 MiB.";
    return;
  }
  if (file.type && !["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
    status.textContent = "Choose a JPEG, PNG, or WebP image.";
    return;
  }
  assignFile(name, file);
  describeExample(null);
  status.textContent = "";
  resetResult();
}

for (const label of form.querySelectorAll(".upload")) {
  const name = label.dataset.slot;
  const input = label.querySelector("input");
  input.addEventListener("change", () => {
    const [file] = input.files;
    input.value = "";
    updateFile(name, file);
  });
  label.addEventListener("dragover", (event) => {
    event.preventDefault();
    label.classList.add("dragging");
  });
  label.addEventListener("dragleave", () => label.classList.remove("dragging"));
  label.addEventListener("drop", (event) => {
    event.preventDefault();
    label.classList.remove("dragging");
    updateFile(name, event.dataTransfer.files[0]);
  });
}
document.addEventListener("dragover", (event) => event.preventDefault());
document.addEventListener("drop", (event) => event.preventDefault());

async function loadExample(key) {
  const [ref, can] = examples[key];
  const generation = ++state.generation;
  submit.disabled = true;
  status.textContent = "Loading fictional example…";
  try {
    const files = await Promise.all([ref, can].map(async (name) => {
      const response = await fetch("/assets/comparison-examples/" + name + ".png");
      if (!response.ok) throw new Error("The fictional example could not be loaded.");
      return new File([await response.blob()], name + ".png", {type: "image/png"});
    }));
    if (generation !== state.generation) return;
    assignFile("reference", files[0]);
    assignFile("candidate", files[1]);
    state.zoom = 1;
    describeExample(key);
    resetResult();
    status.textContent = "";
  } catch (error) {
    if (generation !== state.generation) return;
    showInputs();
    status.textContent = error.message;
  }
}
for (const button of form.querySelectorAll("[data-example]")) {
  button.addEventListener("click", () => loadExample(button.dataset.example));
}
get("clear-pair").onclick = () => {
  for (const url of Object.values(state.urls)) URL.revokeObjectURL(url);
  state.files = {};
  state.urls = {};
  state.zoom = 1;
  status.textContent = "";
  describeExample(null);
  resetResult();
};

function updateOverlays() {
  overlayImages.heat.hidden = !state.accepted || !get("show-heat").checked;
  overlayImages.regions.hidden = !state.accepted || !get("show-regions").checked;
  overlayImages.overlap.hidden = !state.accepted || !get("show-overlap").checked;
  const opacity = Number(get("opacity").value);
  overlayImages.heat.style.opacity = opacity / 100;
  get("opacity-value").value = opacity + "%";
}

function showResult(result) {
  const alignment = result.alignment;
  state.accepted = alignment.supported;
  const decisions = {
    difference_detected: "Document-level difference flagged",
    no_difference_flagged: "No document-level difference flagged",
    inconclusive: "Inconclusive comparison",
  };
  get("comparison-decision").textContent = decisions[result.decision];
  get("comparison-decision").dataset.state = result.decision;
  get("comparison-detail").textContent = state.accepted ? "This decision comes from comparing the document score with the frozen threshold."
    : result.reason === "unsupported_dimensions" ? "The image dimensions are unsupported."
    : "Registration was unreliable. Check the pairing and try a clearer, full-page capture.";
  get("score").textContent = state.accepted ? result.score.toFixed(4) : "Unavailable";
  get("threshold").textContent = result.threshold.toFixed(4);
  get("alignment").textContent = state.accepted ? "Accepted" : "Inconclusive";
  get("overlap").textContent = state.accepted ? (alignment.valid_overlap * 100).toFixed(1) + "%" : "Unavailable";
  get("inliers").textContent = alignment.inliers ?? "Unavailable";
  get("diagnostics").hidden = false;
  get("visual-evidence").hidden = !state.accepted;
  if (state.accepted) {
    get("evidence-detail").textContent = result.region_pixels > 0
      ? "The region overlay highlights local structural differences. Regions can be visible even when the document score stays below the threshold."
      : "No region met the overlay criteria. The heatmap shows structural variation within the valid overlap.";
    const images = result.previews;
    reference.src = images.reference;
    candidate.src = images.aligned_candidate;
    before.querySelector("img").src = images.reference;
    overlayImages.heat.src = images.heatmap;
    overlayImages.regions.src = images.regions;
    overlayImages.overlap.src = images.valid_overlap;
  }
  get("swipe-view").disabled = !state.accepted;
  get("overlays").disabled = !state.accepted;
  updateOverlays();
  setView(false);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (submit.disabled) return;
  const body = new FormData();
  for (const name of ["reference", "candidate"]) body.append(name, state.files[name]);
  resetResult();
  const generation = state.generation;
  submit.disabled = true;
  status.textContent = "Aligning and comparing images…";
  try {
    const response = await fetch("/api/v1/compare", {method: "POST", body, signal: AbortSignal.timeout(40000)});
    const result = await response.json();
    if (generation !== state.generation) return;
    if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "The comparison could not run.");
    showResult(result);
    status.textContent = "";
  } catch (error) {
    if (generation === state.generation) status.textContent = error.name === "TimeoutError" ? "The comparison timed out." : error.message;
  } finally {
    if (generation === state.generation) submit.disabled = false;
  }
});

get("side-by-side").onclick = () => setView(false);
get("swipe-view").onclick = () => setView(true);
get("swipe").oninput = (event) => viewports[1].style.setProperty("--split", event.target.value + "%");
get("overlays").oninput = updateOverlays;
function zoom(amount) {
  state.zoom = Math.min(4, Math.max(1, state.zoom + amount));
  paintCamera();
}
get("zoom-in").onclick = () => zoom(.25);
get("zoom-out").onclick = () => zoom(-.25);
get("reset-view").onclick = () => { state.zoom = 1; paintCamera(); };

for (const viewport of viewports) {
  let drag;
  viewport.addEventListener("pointerdown", (event) => {
    if (state.zoom <= 1 || event.button !== 0) return;
    viewport.setPointerCapture(event.pointerId);
    drag = {id: event.pointerId, x: event.clientX, y: event.clientY};
  });
  viewport.addEventListener("pointermove", (event) => {
    if (!drag || event.pointerId !== drag.id) return;
    state.x += (event.clientX - drag.x) / viewport.clientWidth * 100;
    state.y += (event.clientY - drag.y) / viewport.clientHeight * 100;
    drag.x = event.clientX;
    drag.y = event.clientY;
    paintCamera();
  });
  viewport.addEventListener("lostpointercapture", () => { drag = null; });
  viewport.addEventListener("keydown", (event) => {
    const shift = {ArrowLeft: [-5, 0], ArrowRight: [5, 0], ArrowUp: [0, -5], ArrowDown: [0, 5]}[event.key];
    if (!shift || state.zoom <= 1) return;
    event.preventDefault();
    state.x += shift[0];
    state.y += shift[1];
    paintCamera();
  });
}
loadExample("altered");

