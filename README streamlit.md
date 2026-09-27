# Course Companion — Streamlit version

A single-file Python app — no Node.js, no npm, no frontend build step.

## Files
- `app.py` — the entire website (UI + search + AI answer)
- `requirements.txt` — Python packages needed
- `videos.json` — video titles + links (already filled ones from your
  earlier project can be copied in here — same format)

You also need your own `embeddings.parquet` file — copy it into this same
folder, next to `app.py`.

## Setup (one time)

1. Make sure Ollama is installed and running, with both models pulled:
   ```bash
   ollama pull bge-m3
   ollama pull llama3.2
   ```

2. Install Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy your `embeddings.parquet` file into this folder.

4. If you already filled in your video links in an earlier `videos.json`,
   copy that file here (replacing this one). Otherwise, edit the `video_url`
   fields in this `videos.json` yourself.

## Run it

```bash
streamlit run app.py
```

A browser tab opens automatically at `http://localhost:8501` — the website
is live. No separate backend server, no separate frontend server — this one
command runs everything.

## Notes
- Every time you edit `videos.json`, just refresh the browser tab — Streamlit
  picks up file changes automatically most of the time; if not, stop
  (`Ctrl+C`) and re-run `streamlit run app.py`.
- If searches fail with an Ollama error, check that the Ollama app/service is
  running in the background.
