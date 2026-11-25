# ForgeryLens historical research report

[Current course report on single-image receipt review (PDF)](ForgeryLens_Technical_Report.pdf).

The [single-image follow-up](SINGLE_IMAGE_RESULTS.md) extends this historical report with decoded-image and perceptual candidate checks, source-group calibration, and task-specific MobileNetV3 fine-tuning under a fixed evaluation protocol.

## Abstract

ForgeryLens investigates whether localized receipt changes can be identified from a candidate image, first by itself and then with a trusted reference. The initial whole-image models and subsequent patch and multiple-instance models showed little receipt-level discrimination on AIForge-Doc. Smaller regions sometimes provided useful localization evidence when edit locations were known or a candidate could be ranked against other tiles. That evidence did not yield reliable image-only document decisions. Registered structural similarity (SSIM), a classical two-image method, reached 0.647 AUROC under ordinary simulated acquisition changes and 0.600 under stronger simulated changes. At a Phase E operating point with 4.9% observed false-positive rate, ordinary-regime recall was 6.7%. No independent physical-capture evaluation has been completed.

## Problem and evidence

A small edited field can occupy little of a long receipt. Global resizing can remove local detail, and whole-document pooling can be driven by normal visual variation. We evaluated document classification and localization separately. AUROC measures ranking of positives versus negatives, while pixel IoU measures overlap with the annotated edit. Tile hit rates measure whether at least one selected tile intersects the edit and can improve independently of receipt-level AUROC. A reference-based system has an additional trust assumption: the caller already knows which original document belongs to the candidate. Visual changes can be authorized. Neither task supplies intent or authenticity evidence on its own.

The historical synthetic release trained on procedurally generated fictional receipts. Its selected model reached 0.794 AUROC on its clean synthetic holdout and 0.512 on its shifted synthetic holdout. The hosted historical demo is a curated demonstration of that research release. The external receipt benchmark used AIForge-Doc v2.0.1 paired originals, edited images, and binary masks. The importer filtered to receipts, retained entire source groups, and removed 31 groups connected by byte-identical images. The resulting 2,617 source pairs consisted of 791 train, 228 validation, 262 in-domain CORD/SROIE test, and 1,336 shifted WildReceipt test pairs. The shifted set changes source corpus within one editing generator. The canonical manifest SHA-256 is `3a4723b5bbec3b2fec496836fe48b0fa45a7732437956f1839a7dfce952cfc80`. Exact-byte deduplication does not resolve near-duplicates. Image files and restricted local outputs remain outside Git. [Dataset audit](DATASET_AUDIT.md) · [Model card](MODEL_CARD.md)

## Image-only models and Phase A diagnostics

The initial CNN used grayscale whole-receipt tensors at 112 × 160 and strided convolution blocks. The whole-image head classified pooled features. The local variant combined an input high-pass residual, a spatial evidence map, top-k evidence, and optional global context. The 4,818-parameter whole-image baseline scored 0.501 AUROC on both frozen AIForge holdouts. The 4,890-parameter local teacher scored 0.500 in-domain and 0.504 on WildReceipt, with mean mask IoU of 0.022 and 0.026. The frozen probes confirmed substantial domain dependence. [Frozen benchmark](ForgeryLens_AIForge_Results.md)

An 80-epoch training diagnostic tested 24 paired receipts and checked labels, binary masks, dimensions, gradient norms, and score calibration. The original joint configuration reached 0.568 training AUROC and 0.070 mask IoU. Increasing channel width to 24 and reducing batch size to eight yielded 0.828 training AUROC. The subsequent 128-pair width/batch screen on all 228 development-validation pairs still yielded approximately 0.50. The reduced batch also changed update counts and batch statistics. Capacity and optimization contributed to the original fitting behavior; the controls do not isolate a single causal mechanism. [Phase A](MODEL_FAILURE_ANALYSIS.md)

## Native patch learning, hard negatives, and MIL

**Phase B.** RGB oracle crops centered using the annotation retained more evidence than whole-page resizing. Tight 112-pixel crops averaged 0.574 AUROC across three development-validation seeds, versus 0.531 with 224-pixel context. Oracle metrics use known edit positions. At inference, fixed 112-pixel tiling with stride 84 removed annotation access. A 64-pair validation screen reached 0.520 whole-receipt AUROC with max pooling and 12.5% hit@1. [Phase B](PATCH_EVIDENCE_ANALYSIS.md)

**Phase C.** Same-source background negatives used unedited tiles from both members of the training pair. Adding random off-edit negatives improved hit@1 to 31.3% and hit@3 to 56.3% in the seed-42 validation screen. Model-mined negatives reached 0.521 with top-three pooling, compared with 0.520 for the historical max-pooling screen. At a training-derived threshold the mined model achieved 14.1% recall at 10.9% FPR. Better tile localization did not translate to better receipt-level ranking. [Phase C](HARD_NEGATIVE_ANALYSIS.md)

**Phase D.** Frozen PatchNet embeddings supported fixed pooling, score summaries, attention pooling and spatial aggregation. A shuffled-label control and tile-count baseline checked shortcut behavior. On all 228 validation pairs, max pooling yielded 0.503 AUROC. The three attention seeds ranged from 0.503 to 0.510. The same validation comparison found 0.507 for frozen ImageNet MobileNetV3-Small with attention and 0.500 for the random encoder control. The ImageNet configuration achieved 14.0% recall at 14.0% FPR at its nominal 5% development operating point. Phase D provides no basis for operational image-only verification. [Phase D](MIL_ANALYSIS.md)

## Reference-assisted verification

**Phase E.** The candidate was compared with a trusted copy of the same document. A common raster and ORB feature matching with RANSAC homography estimated image alignment. Geometric checks gated registration. On valid overlap, grayscale photometric normalization preceded an SSIM difference map, with the mean of the highest 0.5% of pixel differences as the document score. Morphology and connected-component filtering produced local regions. A separate 6,090-parameter Siamese model learned paired features from 128 training sources. Variants tested registration and localization loss.

A fixed 64-pair validation calibration subset preceded the 164-pair evaluation subset from the development partition. Exact pairs had no simulated nuisance. Ordinary and difficult regimes independently varied JPEG encoding, geometry, photometry, blur, and noise, with matching distributions across classes. Phase E all-pair ranking retained diagnostic fallback scores on registration failure. Phase F suppresses such scores.

| Ordinary simulation, Phase E development evaluation | AUROC | Pixel IoU |
| --- | ---: | ---: |
| Raw RGB absolute difference | 0.494 | 0.014 |
| Registered RGB difference | 0.552 | 0.044 |
| Registered SSIM | 0.647 | 0.104 |
| Siamese without registration | 0.512 | — |
| Siamese with registration | 0.507 | — |
| Siamese with registration and pixel loss | 0.478 | 0.000 |

Exact-pair registered SSIM reached 1.000 AUROC with 0.619 IoU. Under difficult nuisance it reached 0.600 and 0.051. Ordinary and difficult results bracket sensitivity to simulated acquisition conditions. The exact score reflects unusually well-matched paired imagery. The Siamese variants did not beat the classical baseline. [Phase E](REFERENCE_VERIFICATION_ANALYSIS.md)

**Phase F.** The registration gates and preprocessing are frozen under `registered-ssim-phase-f-v1`. The primary rule flags scores strictly above `0.5850238270653461`, chosen from Phase E calibration at a nominal 5% FPR, with additional frozen 10% and 20% cutoffs. Poor registration or unsupported dimensions result in inconclusive comparisons with no score or evidence maps. Coverage and wrong-reference acceptance are explicitly measured by the protocol. Rejection cannot establish that a pair has the wrong identity. The audit found no rights-cleared, source-disjoint independent physical-capture collection spanning all scenarios, so no independent results are available. [Phase F](REAL_WORLD_VERIFICATION_ANALYSIS.md)

## Conclusions and remaining evidence

The image-only experiments reveal limited and fragile discrimination across the evaluated settings. Reference-assisted SSIM is suitable for explaining alignment, structural residuals and abstention in a research demo. Phase G implements a local workbench and evaluates its software behavior with fictional documents. Broader claims require a predeclared, independently collected set of unchanged recaptures, controlled edits, legitimate revisions, and mismatches. Freeze reference-trust policy and operating points before opening evaluation labels. Report false-positive rates, recall, localization, acceptance of incorrect references, abstention, and source-group uncertainty per acquisition stratum. Handle operational privacy, threat modeling, load and cost separately. [Experiment summary](EXPERIMENT_SUMMARY.md) · [Collection guide](REAL_CAPTURE_COLLECTION_GUIDE.md) · [Deployment guide](DEPLOYMENT_GUIDE.md)

## AI assistance and authorship

Codex assisted with implementation, experiment execution and analysis recorded in the phase logs, the Phase G demonstration and tests, and drafting this synthesis. The recorded measurements are attributed to their original runs. The submitting author must verify the figures, reference terms, and required course disclosure before submission.

