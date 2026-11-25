# Demonstrate ForgeryLens

## Receipt Studio web preview

Open the [Worker URL](https://patient-cake-9531.marboo786.workers.dev/) or follow the [local run instructions](../README.md#receipt-studio-frontend). Compare the original and edited specimen in the hero, then replay a saved specimen in the studio. Use zoom, the tile grid, and the overlay opacity to inspect the model's highest-scoring patch. The interactive charts show published aggregates. The threshold explorer uses development results.

The default export replays recorded results for the bundled fictional specimens. A custom image can be previewed locally. Live scoring requires a separately configured API. [Deployment and API setup](DEPLOYMENT.md)

The build copies `results.json` and `sample-results.json` from [the frontend data directory](../web/frontend/data). The first file records its source paths and digests, including development reliability and risk-coverage results, Pareto and stress summaries, and the frozen historical cohort aggregates. The second file records the bundled specimen hashes and the model results replayed by the browser. The [export scripts](../web/frontend/scripts) require the local research artifacts. Regular frontend builds use the checked-in JSON.

## Frozen receipt detector

From the repository root, with the local research artifacts available, install the [receipt demo environment](../research/requirements-receipt-demo.txt) and start Streamlit. The launch command limits uploads to 8 MB and keeps the service on loopback. Use only images you are authorized to process.

```sh
python -m pip install -r research/requirements-receipt-demo.txt
python -m streamlit run research/receipt_demo.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --server.maxUploadSize 8 --browser.gatherUsageStats false
```

Open [ForgeryLens Receipt Review](http://127.0.0.1:8501). Select a receipt and choose **Analyze receipt**. Read the estimated edit probability, binary confidence, review outcome, and advice together. The highlighted tile shows the model's highest-scoring patch. Expand **Model and routing policy** for the checkpoint hash and cutoffs. The confidence gate deferred all 400 development receipts. On the historical shifted receipts, the full demo prioritized zero, routed six genuine and six edited receipts to the normal queue, and deferred 2,660. An inconclusive result illustrates the intended fallback to manual review. [Model card and evidence](MODEL_CARD.md#aiforge-v2-single-image-receipt-detector)

Startup requires `artifacts/receipt-study-v1/residual-42/fit.json`, `model.pt`, and `calibration.csv`, plus `artifacts/receipt-retrospective-v1/freeze.json`. The application checks the lock and artifact hashes locally before scoring. The development parity test uses the local authorized AIForge assets and is skipped if those assets are absent. The `receipt-demo-v1` verification record is in `artifacts/receipt-demo-v1/verification.json`.

## Historical web demonstrations

Use the [local research setup](history/DEPLOYMENT_GUIDE.md#local-research-demonstration). Open `/studio` for reference comparison. Use **Single-image demo** for the historical synthetic model.

For the historical single-image demo by itself, start from a fresh shell with the research flag unset. Run from the repository root in Bash or PowerShell.

```sh
python -m pip install -r api/requirements.txt
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

Choose **Original specimen** or **Edited specimen** and select **Analyze**. The low-quality specimen illustrates abstention. The curated pair comes from a fictional population outside the test metrics.

## Walkthrough

1. Select **Unchanged** or **Altered total**. The two upload slots show the selected fictional documents before any comparison. Use **Compare documents** to run the frozen policy.
2. Inspect both pages side by side. The viewer zoom controls affect both pages. Drag a zoomed page or focus it and use the arrow keys to pan. **Reset view** returns both pages to fit.
3. When registration succeeds, use **Before / after** and its position slider to sweep between the aligned candidate and reference. Adjust heatmap opacity or toggle the region and valid-overlap layers. The overlays are drawn in the normalized reference raster. Read **Document decision** and **Localized visual evidence** separately.
4. Select **Misaligned** to illustrate abstention. The raw pair remains visible for inspection and the score and overlays are unavailable.
5. Select **Different reference** to illustrate a known pairing mistake. An accepted geometric alignment or a high difference score cannot tell whether both images depict the same document.
6. Use **Clear** to reset both viewers and release their local preview URLs.

| Fictional example | What it illustrates |
| --- | --- |
| Unchanged | An identical source image in both slots. The registered comparison scored below the document threshold. |
| Altered total | A fictional total changes from 47.70 to 147.70. The region overlay highlights the change. The recorded document score is 0.1062 against the frozen threshold of 0.5850, so the decision is **No document-level difference flagged**. |
| Misaligned | A severe crop and rotation of the fictional original. Registration abstained. |
| Different reference | Two deliberately unrelated fictional receipts. The comparator accepted the alignment and flagged a difference. The example defines the identity mismatch independently. A high score cannot establish identity or confirm a valid pairing. |

These examples are original fictional illustrations. They do not estimate accuracy on independent camera captures. When using other examples, both images must be authorized and in JPEG, PNG or WebP format. Full-page captures with visible shared features support registration. Prefer the bundled examples for presentations. [Measured example outcomes](history/EXPERIMENT_SUMMARY.md#phase-g-local-interface-benchmark) · [Research conclusions](history/FINAL_RESEARCH_REPORT.md)

