# Archived Next.js shell

The standalone receipt studio in [web/frontend](../../web/frontend) supersedes this wrapper. The shell previously loaded the legacy studio script and proxied requests to FastAPI.

The legacy HTML and assets remain under `web/public` for the FastAPI routes, the production Docker image, and browser and container QA. The opt-in reference comparison page remains at `web/reference_studio.html` because the reference adapter and its tests load that path. The Next.js receipt studio also imports its fictional specimen PNGs from `web/public`.
