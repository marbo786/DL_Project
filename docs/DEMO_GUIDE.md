# Demonstrate ForgeryLens

Use the [local research setup](DEPLOYMENT_GUIDE.md#local-research-demonstration). Open `/studio` for reference comparison. Use **Single-image demo** for the historical synthetic model.

## Walkthrough

1. Select **Unchanged** or **Altered total**. The two upload slots show the selected fictional documents before any comparison. Use **Compare documents** to run the frozen policy.
2. Inspect both pages side by side. The viewer zoom controls affect both pages. Drag a zoomed page or focus it and use the arrow keys to pan. **Reset view** returns both pages to fit.
3. When registration succeeds, use **Before / after** and its position slider to sweep between the aligned candidate and reference. Adjust heatmap opacity or toggle the region and valid-overlap layers. The overlays are drawn in the normalized reference raster.
4. Select **Misaligned** to illustrate abstention. The raw pair remains visible for inspection and the score and overlays are unavailable.
5. Select **Different reference** to illustrate a known pairing mistake. An accepted geometric alignment or a high difference score cannot tell whether both images depict the same document.
6. Use **Clear** to reset both viewers and release their local preview URLs.

| Fictional example | What it illustrates |
| --- | --- |
| Unchanged | An identical source image in both slots. The registered comparison scored below the document threshold. |
| Altered total | A fictional total changes from 47.70 to 147.70. The actual frozen score stayed below the document threshold. Inspect the highlighted difference in the total area. |
| Misaligned | A severe crop and rotation of the fictional original. Registration abstained. |
| Different reference | Two deliberately unrelated fictional receipts. The comparator accepted the alignment and flagged a difference. Identity was known from example construction. |

These examples are original fictional illustrations. They do not estimate accuracy on independent camera captures. When using other examples, both images must be authorized and in JPEG, PNG or WebP format. Full-page captures with visible shared features support registration. Prefer the bundled examples for presentations. [Measured example outcomes](EXPERIMENT_SUMMARY.md#phase-g-local-interface-benchmark) · [Research conclusions](FINAL_RESEARCH_REPORT.md)

