import shutil
import hashlib
import threading
import time
import uuid
from pathlib import Path

import streamlit as st
from rag_util import answer_question, process_document_to_croma_db

st.set_page_config(page_title="PDF Chat", page_icon="📘", layout="wide", initial_sidebar_state="expanded")

ROOT = Path(__file__).resolve().parent
SESSION_ROOT = ROOT / ".sessions"
SESSION_ROOT.mkdir(exist_ok=True)
SESSION_TTL_SECONDS = 24 * 60 * 60


def format_file_size(size_bytes):
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.0f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def clean_expired_sessions():
    """Best-effort cleanup for browser sessions that ended without a Streamlit event."""
    now = time.time()
    for folder in SESSION_ROOT.iterdir():
        try:
            marker = folder / ".last_seen"
            last_seen = marker.stat().st_mtime if marker.exists() else folder.stat().st_mtime
            if folder.is_dir() and now - last_seen > SESSION_TTL_SECONDS:
                shutil.rmtree(folder, ignore_errors=True)
        except OSError:
            continue


clean_expired_sessions()


@st.cache_resource
def start_session_cleanup():
    def janitor():
        while True:
            time.sleep(15 * 60)
            clean_expired_sessions()

    threading.Thread(target=janitor, daemon=True, name="pdf-chat-session-cleanup").start()
    return True


start_session_cleanup()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#f4f6ff; --muted:#a7b0ca; --line:rgba(153,173,220,.15); --panel:#111a2b; --accent:#665bff; }
.stApp { background:radial-gradient(ellipse at 18% 0%,#19254a 0%,#0b1221 42%,#080d17 100%); color:var(--ink); font-family:'DM Sans',sans-serif; }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:rgba(10,17,31,.82); border-right:1px solid var(--line); }
[data-testid="stSidebar"] > div { padding-top:1.25rem; }
.brand { display:flex; align-items:center; gap:14px; padding:4px 0 22px; }
.brand-icon { width:48px;height:48px;border-radius:15px;display:grid;place-items:center;background:linear-gradient(140deg,#6f67ff,#4439df);font-size:23px;box-shadow:0 10px 30px #5145ee44; }
.brand h1 {font:800 27px Manrope,sans-serif;margin:0;letter-spacing:-1px}.brand h1 span{color:#a98cff}
.brand p {color:var(--muted);margin:2px 0 0;font-size:14px}
.eyebrow {color:#9ba7c5;text-transform:uppercase;letter-spacing:1.3px;font-size:11px;font-weight:700;margin:24px 0 10px}
.doc-card {background:linear-gradient(135deg,#17233a,#111a2a);border:1px solid var(--line);border-radius:18px;padding:16px;margin:8px 0 16px}
.doc-name {font-weight:700;overflow-wrap:anywhere}.doc-sub{color:var(--muted);font-size:13px;margin-top:5px}
.welcome {padding:45px 30px 24px;text-align:center}.welcome-icon{font-size:42px}.welcome h2{font:700 26px Manrope;margin:10px 0}.welcome p{color:var(--muted);max-width:540px;margin:auto;line-height:1.7}
.hint {border:1px solid var(--line);background:#111a2b99;border-radius:14px;padding:14px 16px;color:#c3cce1;font-size:13px;margin-top:12px}
.stChatMessage {background:rgba(20,29,46,.78);border:1px solid var(--line);border-radius:16px;padding:15px 18px;margin-bottom:12px}
.stChatInput textarea {background:#121c2e!important;color:var(--ink)!important;border:1px solid #34425f!important;border-radius:15px!important}
.stButton > button {border-radius:12px;border:1px solid var(--line);font-weight:600}
[data-testid="stFileUploader"] {background:#10192a;border:1px dashed #46547a;border-radius:16px;padding:8px}
div[data-testid="stStatusWidget"] {visibility:hidden}
</style>
""", unsafe_allow_html=True)

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
    (SESSION_ROOT / st.session_state.session_id).mkdir(parents=True, exist_ok=True)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document" not in st.session_state:
    st.session_state.document = None
if "upload_widget_key" not in st.session_state:
    st.session_state.upload_widget_key = 0

session_dir = SESSION_ROOT / st.session_state.session_id
session_dir.mkdir(parents=True, exist_ok=True)
(session_dir / ".last_seen").touch()

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-icon">📄</div><div><h1>PDF <span>Chat</span></h1><p>Ask questions. Get trusted answers.</p></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Your documents</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=False, label_visibility="collapsed", key=f"pdf_upload_{st.session_state.upload_widget_key}")
    if uploaded is not None:
        content_hash = hashlib.sha256(uploaded.getvalue()).hexdigest()
        if st.session_state.document and st.session_state.document["hash"] != content_hash:
            shutil.rmtree(session_dir / "vectorstore", ignore_errors=True)
            st.session_state.messages = []
            old_file = session_dir / Path(st.session_state.document["name"]).name
            old_file.unlink(missing_ok=True)
            st.session_state.document = None
        if not st.session_state.document:
            stored_file = session_dir / Path(uploaded.name).name
            with st.spinner("Reading and indexing your PDF…"):
                try:
                    stored_file.write_bytes(uploaded.getvalue())
                    process_document_to_croma_db(str(stored_file), persist_directory=str(session_dir / "vectorstore"))
                    st.session_state.document = {"name": uploaded.name, "size": uploaded.size, "hash": content_hash}
                    st.session_state.messages = []
                    st.success("PDF ready to chat")
                except Exception as exc:
                    stored_file.unlink(missing_ok=True)
                    st.error(f"Could not process this PDF: {exc}")
    if st.session_state.document:
        doc = st.session_state.document
        st.markdown(f'<div class="doc-card"><div class="doc-name">📕 {doc["name"]}</div><div class="doc-sub">{format_file_size(doc["size"])} · Indexed for this session</div></div>', unsafe_allow_html=True)
        if st.button("✕  Remove document", use_container_width=True):
            shutil.rmtree(session_dir, ignore_errors=True)
            st.session_state.document = None
            st.session_state.messages = []
            session_dir.mkdir(parents=True, exist_ok=True)
            st.session_state.upload_widget_key += 1
            st.rerun()
    st.markdown('<div class="eyebrow">Tips</div><div class="hint">💡 Upload a PDF, then ask questions in plain language. Answers are based on your document.</div>', unsafe_allow_html=True)
    st.markdown('<div class="hint">🔒 Your files are stored in an isolated session folder and cleaned up after the session expires.</div>', unsafe_allow_html=True)

st.markdown('<div class="brand"><div class="brand-icon">✦</div><div><h1>PDF <span>Chat</span></h1><p>Ask questions. Get trusted answers.</p></div></div>', unsafe_allow_html=True)

if not st.session_state.document:
    st.markdown('<div class="welcome"><div class="welcome-icon">📚</div><h2>Your documents, ready to talk</h2><p>Upload a PDF using the panel on the left. I’ll index its contents so you can ask questions and explore the details.</p></div>', unsafe_allow_html=True)
    st.info("Start by uploading a PDF in the sidebar.")
else:
    doc = st.session_state.document
    st.markdown(f'<div class="doc-card">📕 &nbsp; <strong>{doc["name"]}</strong><div class="doc-sub">Ready for questions · {format_file_size(doc["size"])}</div></div>', unsafe_allow_html=True)
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "✨"):
            st.markdown(message["content"])
    if not st.session_state.messages:
        st.markdown('<div class="hint">Try asking for a summary, a definition, or an explanation of a specific section.</div>', unsafe_allow_html=True)
    prompt = st.chat_input("Ask a question about this document…")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="✨"):
            with st.spinner("Finding the answer in your PDF…"):
                try:
                    answer = answer_question(prompt, persist_directory=str(session_dir / "vectorstore"))
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as exc:
                    st.error(f"I couldn’t answer that right now: {exc}")
