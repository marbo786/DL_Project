# ForgeryLens model card

**Release scope** Synthetic receipt research and software demonstration. The factory loads the validation-selected model from `release/model.npz`. Measured classification, calibration, localization, model size, cost and held-out shift results are stored under `release/`.

**Input and output** The model converts an EXIF-oriented document image to grayscale and resizes it to 112 × 160 pixels. It produces a document logit and coarse evidence map. Candidate regions are normalized to the original displayed image. Confidence uses validation-fitted affine logit calibration, and uncertainty is twice the smaller binary class probability.

**Training and evidence** Fictional source documents were paired with localized edit variants and masks. Source groups were assigned independently to training, validation, clean test and shifted test. The synthetic training edit changes the local tonal appearance and redraws a fictional amount. The shifted operator modifies quality around existing text. This operator shift is materially narrower than a real generator shift.

**Operating point** The temperature and acceptance threshold are fitted on validation only. The packaged synthetic model scores the normalized demo specimens. It abstains on arbitrary uploads and on the known low-quality specimen. A model trained with an independently authorized manifest can be exported through the same interface. Verify its scope and calibration before use.

**Failure modes and usage limits** Synthetic results do not establish skill on scans, camera captures, real receipts, or new editing systems. Resizing can suppress very small edits. Candidate regions reflect model activations and can miss or overestimate affected areas. This tool cannot establish authenticity, intent, identity, or transaction validity.

**Data handling** The shipped service processes encoded images in memory and retains no upload history. Hosting providers may collect routine request metadata. No external documents were bundled. The known demo specimens are fictional.

**Attribution** Training and evaluation source and measurements are included. AI assistance generated the implementation and report from measured outputs.