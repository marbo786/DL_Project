"""Generate the technical report from a completed, measured release."""

import argparse
from html import escape
import json
from pathlib import Path

import fitz

LABELS = {
    "baseline": "Whole-image baseline", "local": "Selected local model",
    "fp16_storage": "FP16 weight storage", "distilled_student": "Distilled student",
    "grayscale_only": "Grayscale only", "average_pooling": "Average pooling",
    "no_global_context": "No global context", "classification_only": "Classification loss only",
}


def read(directory, filename):
    return json.loads((directory / f"{filename}.json").read_text(encoding="utf-8"))


def metric(value, decimals=3):
    return "-" if value is None else f"{value:.{decimals}f}"


def build(directory, destination):
    summary = read(directory, "summary")
    ablations = read(directory, "ablations")
    robust = read(directory, "robustness")
    cal = read(directory, "calibration")
    export = read(directory, "export")
    choice = summary["selected_model"]
    name = LABELS[choice]
    scope = summary["scope"]
    clean, shifted = robust["clean"], robust["shifted"]
    body = []

    def text(value, style="BodyText"):
        tag = {"Title": "h1", "Heading2": "h2"}.get(style, "p")
        body.append(f"<{tag}>{value}</{tag}>")

    def table(header, rows, widths):
        def cells(values, tag):
            return "".join(f"<{tag}>{escape(str(value))}</{tag}>" for value in values)
        column_html = "".join(f"<col width='{width / sum(widths):.2%}'>" for width in widths)
        contents = "".join(f"<tr>{cells(row, 'td')}</tr>" for row in rows)
        body.append(f"<table>{column_html}<tr>{cells(header, 'th')}</tr>{contents}</table>")

    text("ForgeryLens", "Title")
    text("Technical evaluation of localized receipt edits", "Heading2")
    text(f"Measured run {escape(summary['run_id'])}. Evaluation scope: {escape(scope)}.")
    text(f"The selected model was the {name.lower()}. On the clean test split, macro F1 was "
         f"<b>{metric(clean['macro_f1'])}</b> and manipulation recall was "
         f"<b>{metric(clean['manipulated_recall'])}</b>. With a held-out edit operator, macro F1 was "
         f"<b>{metric(shifted['macro_f1'])}</b> and recall was "
         f"<b>{metric(shifted['manipulated_recall'])}</b>. These measurements describe a fictional benchmark.")
    text("Dataset and model", "Heading2")
    text(f"The benchmark used {summary['training_examples']} training images and {summary['validation_examples']} "
         "validation images. Source-grouped pairs contain one original and one localized edit. "
         "The training edit changes the local background and redraws one amount field. The shifted test "
         "instead modifies local blur and contrast. The same document's variants never cross partitions. "
         "All examples were programmatically generated for this project.")
    text("The baseline reduces the full image to a global score. Local variants use a small convolutional "
         "spatial head, optional residual input and context, and mean or top-k pooling. The localization "
         "loss supervises a downsampled binary mask. Mask IoU measures overlap on manipulated images.")
    text("Accuracy and efficiency", "Heading2")
    table(["Model", "Clean F1", "Shifted F1", "Parameters", "Checkpoint", "Median CPU"],
          [[LABELS[key], metric(row["test"]["macro_f1"]), metric(row["shifted_test"]["macro_f1"]),
            f'{row["params"]:,}', f'{row["checkpoint_bytes"] / 1024:.1f} KiB',
            f'{row["timing"]["median_ms"]:.2f} ms'] for key, row in summary["models"].items()],
          [44, 22, 25, 26, 28, 27])
    dominated = [LABELS[key] for key, row in summary["models"].items() if row["pareto_dominated"]]
    text("The FP16 variant compresses storage and runs with FP32 arithmetic. The smaller student learns "
         "from the teacher's scores and evidence maps. Validation F1 and median CPU timing "
         "identified the following dominated entries in this run: " + (", ".join(dominated) if dominated else "none") + ".")
    text(f"The deployed NumPy model occupies {export['export_size_bytes'] / 1024:.1f} KiB. Its warm CPU "
         f"forward-pass median was {export['batch_1_cpu_median_ms']:.2f} ms, with a "
         f"{export['batch_1_cpu_p95_ms']:.2f} ms p95. Parity with PyTorch was verified on "
         f"{export['parity_examples']} additional generated inputs.")
    body.append("<div style='page-break-before: always'></div>")
    text("Ablations and robustness", "Title")
    text("Validation ablations report mean and sample standard deviation over seeds "
         + ", ".join(map(str, summary["seeds"])) + ". Training used a fixed epoch budget and validation checkpoint selection.")
    aggregate = ablations["validation_aggregate"]

    def mean_std(row):
        return "-" if row is None else f'{row["mean"]:.3f} +/- {row["std"]:.3f}'

    table(["Variant", "Macro F1", "ROC-AUC", "Mask IoU"],
          [["Full local model" if key == "local" else LABELS[key], mean_std(row.get("macro_f1")), mean_std(row.get("roc_auc")), mean_std(row.get("mean_mask_iou"))]
           for key, row in aggregate.items()], [58, 38, 38, 38])
    text("Stress tests", "Heading2")
    table(["Condition", "Macro F1", "Edit recall", "Mask IoU"],
          [[key.replace("_", " ").title(), metric(row["macro_f1"]), metric(row["manipulated_recall"]), metric(row.get("mean_mask_iou"))]
           for key, row in (("clean", clean), *robust["stress"].items(), ("held-out edit", shifted))], [58, 38, 38, 38])
    text("Calibration and abstention", "Heading2")
    text(f"Affine logit calibration used validation data only. The fitted temperature was {cal['temperature']:.3f}. "
         f"Clean-test ECE moved from {cal['test_before']['ece']:.3f} to {cal['test_after']['ece']:.3f}. "
         f"The fitted bias was {cal.get('bias', 0):.3f}. The confidence cutoff was {cal['confidence_threshold']:.3f}. Clean-test coverage was "
         f"{cal['test_after']['selective']['coverage']:.1%}; shifted coverage was "
         f"{cal['shifted_after']['selective']['coverage']:.1%}. Coverage counts accepted predictions. "
         "Error conditional on acceptance is undefined when coverage is zero.")
    body.append("<div style='page-break-before: always'></div>")
    text("Failure cases and model card", "Title")
    text("The following rows are the highest-confidence shifted errors from the frozen model. "
         "The artifact retains original split indices for reproducibility.")
    table(["Shifted index", "Expected", "Predicted", "Edit probability", "Mask IoU"],
          [[row["shifted_test_index"], "Edited" if row["expected"] else "Original",
            "Edited" if row["predicted"] else "Original", metric(row["manipulation_probability"]), metric(row["mask_iou"])]
           for row in robust["shifted_failure_cases"]], [35, 30, 30, 42, 35])
    text("Held-out edits can preserve background texture and change character sharpness only. "
         "A detector trained on stronger tonal seams may miss those edits. This is a plausible mechanism, "
         "with effect sizes reported in the stress tests. The current experiment does not identify causal image features.")
    text("Intended use and constraints", "Heading2")
    text("Use the release for demonstrations on fictional specimens. A synthetic-trained checkpoint deliberately "
         "abstains on unfamiliar uploads. Scores and overlays on supported specimens are model evidence for "
         "inspection only. They cannot certify document authenticity, infer intent, or recover original content.")
    text("A production study requires authenticated dataset access, verified source identities across datasets, "
         "independent genuine and edited samples, generator and scanner diversity, and calibration measured on "
         "the intended population. The data audit records candidate benchmarks and access requirements.")
    text("Reproduction", "Heading2")
    text("Run research.train with 12 epochs, 144 synthetic source groups and two CPU threads, then run "
         "research.export. Inspect release/summary.json, ablations.json, robustness.json, calibration.json "
         "and export.json. Model selection and cutoffs are frozen before test metrics are computed.")
    text("Dataset references", "Heading2")
    text("Scam AI. AIForge-Doc v1 and v2 dataset cards. Hugging Face, Scam-AI/AIForge-Doc-v1 and "
         "Scam-AI/AIForge-Doc-v2. Qu et al. Towards Robust Tampered Text Detection in Document Image. CVPR 2023.")
    text("AI assistance generated the project implementation and report from programmatically measured outputs.")
    css = """
    body { font-family: sans-serif; color: #172b33; font-size: 10.1pt; line-height: 1.35; }
    h1 { font-size: 28pt; line-height: 1.1; margin: 0 0 12pt; color: #172b33; }
    h2 { font-size: 14pt; color: #36535f; margin: 14pt 0 6pt; }
    p { margin: 0 0 9pt; }
    table { border-collapse: collapse; width: 100%; margin: 7pt 0 12pt; font-size: 9pt; }
    th, td { text-align: left; padding: 6pt; border-bottom: 1px solid #d0dadd; }
    th { font-weight: bold; color: #36535f; }
    """
    page = fitz.paper_rect("a4")
    box = fitz.Rect(53, 48, page.width - 53, page.height - 55)
    story = fitz.Story("".join(body), user_css=css)
    writer = fitz.DocumentWriter(str(destination))
    more = True
    while more:
        device = writer.begin_page(page)
        more, _ = story.place(box)
        story.draw(device)
        writer.end_page()
    writer.close()
    with fitz.open(destination) as document:
        document.set_metadata({"title": "ForgeryLens Technical Evaluation", "author": "ForgeryLens Project"})
        for index, current in enumerate(document, start=1):
            y = current.rect.height - 32
            current.draw_line((53, y - 10), (current.rect.width - 53, y - 10), color=(.72, .78, .81))
            current.insert_text((53, y), "ForgeryLens / Technical evaluation", fontsize=8, color=(.2, .32, .38))
            current.insert_text((current.rect.width - 61, y), str(index), fontsize=8, color=(.2, .32, .38))
        document.saveIncr()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    build(args.release, args.destination)
