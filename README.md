# ForgeryLens

ForgeryLens is an AI341 study of edit detection from a single receipt image using AIForge-Doc v2. The frozen MobileNetV3 model scores image tiles and highlights its highest-scoring region.

**Research prototype. The model performs near chance. Do not use it for real reimbursement or fraud decisions.**

| Evaluation cohort | AUROC | Underlying evidence |
| --- | ---: | --- |
| CORD/SROIE development | 0.5126 | `results/receipt-study-v1/residual-42/metrics.json` |
| WildReceipt retrospective shift | 0.5015 | `results/receipt-retrospective-v1/shifted_test.json` |
| XFUND held-back forms | 0.5007 | `results/receipt-xfund-v1/metrics.json` |

The [model card](docs/MODEL_CARD.md) explains the cohorts, frozen policy, and limitations. AUROC values are rounded from the JSON in [the evidence directory](results/README.md). The public frontend ships the [published aggregate export](web/frontend/data/results.json).

## Try the demo

Run this from the repository root in Bash or PowerShell with Node.js 20.9 or newer:

```sh
npm --prefix web/frontend run demo
```

Open [localhost:3000](http://localhost:3000). The studio replays saved outputs for fictional specimens. Custom images use local preview mode by default.

[Course report (PDF)](docs/ForgeryLens_Technical_Report.pdf) · [Documentation index](docs/README.md) · [Contributors](docs/AUTHORSHIP_AND_AI_DISCLOSURE.md)

## Licensing

Project code is [MIT licensed](LICENSE). [AIForge-Doc v2](https://huggingface.co/datasets/Scam-AI/AIForge-Doc-v2#licence) uses [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). ForgeryLens treats AIForge-Doc-derived weights as inheriting those noncommercial share-alike terms and keeps them local. Source datasets and pretrained weights may carry additional terms. See the [checkpoint provenance](docs/SINGLE_IMAGE_REPRODUCTION.md#pretrained-checkpoint-provenance).
