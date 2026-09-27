import json
import re
from pathlib import Path
import gdown

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ── Config ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
EMBEDDINGS_PATH = BASE_DIR / "embeddings.parquet"
VIDEOS_JSON_PATH = BASE_DIR / "videos.json"

OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "bge-m3"
GENERATION_MODEL = "llama3.2"
TOP_K = 5

st.set_page_config(page_title="Course Companion", page_icon="🎓", layout="centered")


# ── Helpers ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embeddings() -> pd.DataFrame | None:
    if not EMBEDDINGS_PATH.exists():
        return None
    frame = pd.read_parquet(EMBEDDINGS_PATH)
    required = {"title", "number", "start", "end", "text", "embedding"}
    if not required.issubset(frame.columns):
        return None
    frame = frame.copy()
    frame["embedding_vector"] = frame["embedding"].apply(lambda v: np.asarray(v, dtype=np.float32))
    matrix = np.vstack(frame["embedding_vector"].to_numpy())
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    frame["normalized_embedding"] = list(matrix / np.maximum(norms, 1e-12))
    return frame


@st.cache_resource(show_spinner=False)
def load_videos() -> dict:
    if not VIDEOS_JSON_PATH.exists():
        return {}
    with VIDEOS_JSON_PATH.open("r", encoding="utf-8") as f:
        videos = json.load(f)
    return {v["number"]: v for v in videos}


def extract_drive_id(url: str) -> str | None:
    match = re.search(r"(?:id=|/d/)([a-zA-Z0-9_-]{20,})", url)
    return match.group(1) if match else None


@st.cache_data(show_spinner=False, ttl=3600, max_entries=6)
def fetch_drive_video_bytes(file_id: str) -> bytes | None:
    """Downloads a Google Drive file, bypassing the 'can't scan for viruses'
    warning that normally blocks large files from playing directly."""
    session = requests.Session()
    url = "https://drive.google.com/uc?export=download"
    try:
        response = session.get(url, params={"id": file_id}, stream=True, timeout=60)
        token = None
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                token = value
        if token is None:
            # Newer Drive pages embed the confirm token in the HTML instead of a cookie
            match = re.search(r"confirm=([0-9A-Za-z_-]+)", response.text)
            if match:
                token = match.group(1)
        if token:
            response = session.get(
                url, params={"id": file_id, "confirm": token}, stream=True, timeout=60
            )
        content_type = response.headers.get("Content-Type", "")
        if "text/html" in content_type:
            return None  # still hit a warning/HTML page instead of the actual video
        return response.content
    except requests.exceptions.RequestException:
        return None


def format_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


def embed_query(query: str) -> np.ndarray | None:
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": query},
            timeout=60,
        )
        r.raise_for_status()
        embedding = r.json().get("embedding")
        if not embedding:
            return None
        vector = np.asarray(embedding, dtype=np.float32)
        return vector / max(float(np.linalg.norm(vector)), 1e-12)
    except requests.exceptions.RequestException:
        return None


def build_context(matches: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] {m['title']} (video {m['number']}, {m['start']:.1f}s-{m['end']:.1f}s)\n{m['text']}"
        for i, m in enumerate(matches, start=1)
    )


def generate_answer(query: str, matches: list[dict]) -> str | None:
    prompt = f"""You are Course Companion, a precise teaching assistant for a video course.
Answer the student's question using only the transcript excerpts below. Explain which lecture sections are relevant and give practical guidance. If the excerpts do not fully answer the question, say what is missing instead of inventing facts. Keep the response concise and readable with short paragraphs.

Student question: {query}

Transcript excerpts:
{build_context(matches)}"""
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": GENERATION_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip() or None
    except requests.exceptions.RequestException:
        return None


def search(query: str, frame: pd.DataFrame, videos: dict) -> tuple[str | None, list[dict], str | None]:
    query_vector = embed_query(query)
    if query_vector is None:
        return None, [], "Could not reach Ollama for embeddings. Make sure Ollama is running."

    matrix = np.vstack(frame["normalized_embedding"].to_numpy())
    scores = matrix @ query_vector
    top_indices = np.argsort(scores)[::-1][:TOP_K]

    matches = []
    for idx in top_indices:
        row = frame.iloc[int(idx)]
        video = videos.get(int(row["number"]), {})
        matches.append(
            {
                "title": str(row["title"]),
                "number": int(row["number"]),
                "start": float(row["start"]),
                "end": float(row["end"]),
                "text": str(row["text"]),
                "video_url": video.get("video_url", ""),
            }
        )

    answer = generate_answer(query, matches)
    if answer is None:
        return None, matches, "Could not reach Ollama for the answer. Make sure Ollama is running."
    return answer, matches, None


# ── Custom styling ──────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stApp { background-color: #0d1014; color: #e6e8eb; }
    .block-container { padding-top: 3rem; max-width: 780px; }
    .badge {
        display: inline-block; background: rgba(99,102,241,0.15); color: #a5b4fc;
        padding: 2px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 600;
        margin-right: 8px;
    }
    .timestamp { color: #94a3b8; font-size: 0.8rem; }
    .result-card {
        background: #161a1f; border: 1px solid #242a31; border-radius: 14px;
        padding: 16px 18px; margin-bottom: 12px;
    }
    .answer-box {
        background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.2);
        border-radius: 14px; padding: 18px; margin-bottom: 24px; line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ─────────────────────────────────────────────────────────────
videos = load_videos()
with st.sidebar:
    st.markdown("### 📚 All Lectures")
    if videos:
        for number in sorted(videos.keys()):
            st.markdown(f"**{number}.** {videos[number]['title']}")
    else:
        st.caption("No lectures found in videos.json")

# ── Main UI ─────────────────────────────────────────────────────────────
st.markdown("## 🎓 Course Companion")
st.caption("Ask anything from the course — I'll find exactly which lecture covers it, down to the second.")

frame = load_embeddings()
if frame is None:
    st.error(
        "⚠️ `embeddings.parquet` not found or missing required columns. "
        "Place your embeddings file next to this app and restart."
    )
    st.stop()

query = st.text_input("", placeholder="Ask anything from the course...", label_visibility="collapsed")
search_clicked = st.button("Ask", type="primary", use_container_width=True)

if search_clicked and query.strip():
    with st.spinner("Searching lectures and generating answer..."):
        answer, matches, error = search(query.strip(), frame, videos)

    if error:
        st.error(f"⚠️ {error}")
    else:
        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

        for i, match in enumerate(matches):
            with st.container():
                st.markdown(
                    f"""
                    <div class="result-card">
                        <span class="badge">Video #{match['number']}</span>
                        <span class="timestamp">⏱ {format_time(match['start'])} – {format_time(match['end'])}</span>
                        <h4 style="margin: 8px 0 4px 0;">{match['title']}</h4>
                        <p style="color:#94a3b8; font-size:0.88rem;">{match['text'][:220]}{'...' if len(match['text']) > 220 else ''}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if match["video_url"]:
                    with st.expander(f"▶️ Watch this part (video #{match['number']})"):
                        drive_id = extract_drive_id(match["video_url"])
                        if drive_id:
                            with st.spinner("Loading video from Google Drive..."):
                                video_bytes = fetch_drive_video_bytes(drive_id)
                            if video_bytes:
                                st.video(video_bytes, start_time=int(match["start"]))
                            else:
                                st.error(
                                    "Couldn't load this video from Google Drive "
                                    "(it may be too large or restricted). "
                                    f"You can open it directly: {match['video_url']}"
                                )
                        else:
                            st.video(match["video_url"], start_time=int(match["start"]))
                else:
                    st.caption(f"No video link configured yet for video #{match['number']} — add one in videos.json")
elif search_clicked:
    st.warning("Please type a question first.")
