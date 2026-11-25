# ForgeryLens release-readiness checklist

## Receipt Studio release

| Check | Status |
| --- | --- |
| Static export, keyboard controls, reduced motion and static content without JavaScript | Passed locally. Playwright covered 1440, 768 and 390 px in both themes. |
| Local Lighthouse | Desktop 99 performance and 100 accessibility. Mobile 83 performance and 100 accessibility. Mobile performance remains below the 90 target. |
| Mobile scrolling | Emulated Chromium checks passed. Physical-phone verification is pending. |
| Cloudflare | Production deployment and hosted smoke test have yet to be verified. |
| Live image scoring | Requires an authorized backend with the frozen artifacts and the deployed frontend origin allowed. |

After deployment, load the [Worker](https://patient-cake-9531.marboo786.workers.dev/) in dark and light mode at desktop and mobile widths. Replay a saved specimen, inspect the tile overlay, and try an unsupported PDF. Open interactive charts and adjust the development threshold. With JavaScript disabled, check that results, method and limits remain readable.

[Archived Phase H container gate](history/PHASE_H_RELEASE.md)
