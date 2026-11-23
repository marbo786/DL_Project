const form = document.querySelector("#comparison-form");
const button = form.querySelector("button");
const status = document.querySelector("#comparison-status");
const result = document.querySelector("#comparison-result");
const decision = document.querySelector("#comparison-decision");
const detail = document.querySelector("#comparison-detail");
const images = document.querySelector("#comparison-images");
const titles = {
  difference_detected: "Visible difference flagged",
  no_difference_flagged: "No difference exceeded the threshold",
  inconclusive: "Inconclusive comparison",
};
const previews = {
  reference: "Trusted reference",
  aligned_candidate: "Aligned candidate",
  heatmap: "Structural difference heatmap",
  region_overlay: "Candidate with difference regions",
  valid_overlap: "Valid comparison area",
};

function renderComparison(data) {
  decision.textContent = titles[data.decision];
  if (!data.alignment.supported) {
    detail.textContent = data.reason === "unsupported_dimensions"
      ? "The image dimensions are unsupported."
      : "The images could not be aligned reliably. Check that the reference is correct and both captures show the full document.";
  } else {
    const {inliers, valid_overlap: overlap} = data.alignment;
    detail.textContent = `SSIM difference score ${data.score.toFixed(3)} · Frozen threshold ${data.threshold.toFixed(3)} · ${inliers} alignment inliers · ${(overlap * 100).toFixed(1)}% valid overlap · ${data.inference_ms.toFixed(0)} ms. A low score cannot verify authenticity.`;
    for (const [key, title] of Object.entries(previews)) {
      const figure = document.createElement("figure");
      const caption = document.createElement("figcaption");
      const image = document.createElement("img");
      image.src = data.previews[key];
      image.alt = title;
      caption.textContent = title;
      figure.append(caption, image);
      images.append(figure);
    }
  }
  result.hidden = false;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const body = new FormData(form);
  button.disabled = true;
  status.textContent = "Comparing images…";
  result.hidden = true;
  images.replaceChildren();
  try {
    const response = await fetch("/api/v1/compare", {method: "POST", body});
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Comparison could not run.");
    renderComparison(data);
    status.textContent = "";
  } catch (error) {
    status.textContent = error.message;
  } finally {
    button.disabled = false;
  }
});
