# Screen recording script (< 5 minutes)

Claude can't record a screen video itself, so here's a tight script you can follow to record
the "code change -> deployed prediction" demo required for submission. Suggested tool: QuickTime
(Mac, built in) or OBS Studio (free, cross-platform).

**Before recording:** have a terminal, your editor, the GitHub repo's Actions tab, and a
browser tab for `http://localhost:8000/docs` (FastAPI's built-in Swagger UI) all ready to
switch between.

1. **(0:00–0:30) Show the running pipeline end to end, briefly.**
   Terminal: `git log --oneline -5` and `ls` the repo — narrate the M1–M5 layout in one breath.

2. **(0:30–1:15) Make a small code change and push it.**
   Edit something trivial but visible — e.g. bump the API `version=` string in
   `src/api/main.py`, or tweak a log message. `git add -A && git commit -m "demo: bump API
   version" && git push`.

3. **(1:15–2:15) Show CI running.**
   Switch to the GitHub repo's **Actions** tab, open the triggered "CI - Test, Build & Publish
   Image" run, and let the viewer see: checkout -> install deps -> pytest passing -> Docker
   build -> push to GHCR. Point at the green checkmarks; no need to wait for it to fully
   finish on camera — you can speed through or cut here if it's slow.

4. **(2:15–3:00) Show CD deploying it.**
   Open the "CD - Deploy & Smoke Test" run that triggers after CI succeeds on `main`. Show the
   "Deploy with Docker Compose" and "Run smoke tests" steps passing.

5. **(3:00–4:00) Hit the live service.**
   Locally: `docker compose up -d` (or point at wherever it's actually running), then:
   - `curl http://localhost:8000/health`
   - Open `http://localhost:8000/docs`, expand `POST /predict`, upload a cat or dog photo,
     execute, and show the returned label + probabilities live.

6. **(4:00–4:45) Show monitoring.**
   `curl http://localhost:8000/metrics | head -20` to show request/latency counters, then run
   `python scripts/simulate_monitoring.py --n-per-class 10` and show the printed
   accuracy/latency summary.

7. **(4:45–5:00) Wrap.**
   One sentence: "That's code push -> CI test & build -> CD deploy -> live prediction ->
   monitored, all with open-source tooling."

Keep narration terse — the goal is showing the pipeline actually working, not explaining
MLOps theory on camera.
