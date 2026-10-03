"""
Text Summarization App
Summarizes long text, .txt or .pdf files using BART models (abstractive
summarization). Long documents are split into chunks, summarized, and merged.
Runs locally. No API key needed.
"""
import streamlit as st
from pypdf import PdfReader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

st.set_page_config(page_title="Text Summarizer", page_icon="📝", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --accent: #4f46e5;
    --accent-dark: #4338ca;
    --accent-soft: #eef2ff;
    --teal-soft: #f0fdfa;
    --teal-border: #99f6e4;
    --text: #1e293b;
    --muted: #64748b;
    --border: #e2e8f0;
    --card: rgba(255, 255, 255, 0.85);
    --shadow: 0 1px 3px rgba(15, 23, 42, 0.06), 0 8px 24px rgba(15, 23, 42, 0.05);
    --shadow-hover: 0 4px 8px rgba(15, 23, 42, 0.07), 0 14px 32px rgba(15, 23, 42, 0.08);
}

/* ---------- Base ---------- */
html, body, [class*="css"], .stApp, .stMarkdown, p, label, input, textarea, button {
    font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif !important;
}
.stApp {
    background: linear-gradient(160deg, #f8fafc 0%, #eef2ff 55%, #f1f5f9 100%);
    color: var(--text);
}
[data-testid="stHeader"] { background: transparent; }
.block-container {
    max-width: 1150px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

/* ---------- Headings & text ---------- */
h1 {
    text-align: center;
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em;
    padding-bottom: 0.2rem !important;
    background: linear-gradient(90deg, #4f46e5 0%, #0ea5e9 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}
h2, h3 {
    color: var(--text);
    font-weight: 700 !important;
    letter-spacing: -0.01em;
}
.block-container > div > div > div > div[data-testid="stElementContainer"] > div[data-testid="stMarkdownContainer"] > p:only-child {
    color: var(--muted);
}
p { line-height: 1.7; color: var(--text); }
[data-testid="stMarkdownContainer"] p { font-size: 1rem; }

/* Subtitle: first paragraph under the title */
[data-testid="stMainBlockContainer"] [data-testid="stElementContainer"]:nth-child(2) [data-testid="stMarkdownContainer"] p {
    text-align: center;
    font-size: 1.1rem;
    color: var(--muted);
    max-width: 640px;
    margin: 0 auto 1.5rem auto;
}

/* Captions */
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
    color: var(--muted) !important;
    font-size: 0.85rem !important;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid var(--border);
    box-shadow: 4px 0 24px rgba(15, 23, 42, 0.04);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
[data-testid="stSidebar"] h2 {
    font-size: 1.3rem !important;
    padding-bottom: 0.75rem;
    margin-bottom: 0.5rem;
    border-bottom: 2px solid var(--accent-soft);
}
[data-testid="stSidebar"] [data-testid="stElementContainer"] {
    margin-bottom: 0.5rem;
}
[data-testid="stSidebar"] label p {
    font-weight: 600;
    color: var(--text);
    font-size: 0.9rem;
}

/* ---------- Radio buttons ---------- */
[data-testid="stRadio"] label p { font-weight: 500; }
[data-testid="stRadio"] [role="radiogroup"] { gap: 0.5rem; }
[data-testid="stRadio"] [role="radiogroup"] > label {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.45rem 0.9rem;
    transition: all 0.2s ease;
    cursor: pointer;
}
[data-testid="stRadio"] [role="radiogroup"] > label:hover {
    border-color: var(--accent);
    background: var(--accent-soft);
}
[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) {
    border-color: var(--accent);
    background: var(--accent-soft);
    box-shadow: 0 0 0 1px var(--accent) inset;
}

/* ---------- Select box ---------- */
[data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    transition: all 0.2s ease;
}
[data-baseweb="select"] > div:hover { border-color: var(--accent) !important; }
[data-baseweb="select"] > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15) !important;
}

/* ---------- Text area ---------- */
[data-testid="stTextArea"] label p { font-weight: 600; }
[data-testid="stTextArea"] textarea {
    background: #ffffff !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 1rem 1.15rem !important;
    font-size: 0.98rem;
    line-height: 1.7;
    color: var(--text);
    box-shadow: var(--shadow);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stTextArea"] textarea::placeholder { color: #94a3b8; }
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.12), var(--shadow) !important;
    outline: none;
}
[data-testid="stTextArea"] > div > div,
[data-testid="stTextArea"] [data-baseweb="textarea"],
[data-testid="stTextArea"] [data-baseweb="base-input"] {
    border: none !important;
    background: transparent !important;
    border-radius: 14px !important;
}

/* ---------- File uploader ---------- */
[data-testid="stFileUploader"] label p { font-weight: 600; }
[data-testid="stFileUploaderDropzone"] {
    background: #ffffff;
    border: 2px dashed #c7d2fe;
    border-radius: 14px;
    padding: 1.5rem;
    transition: all 0.2s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent);
    background: var(--accent-soft);
}
[data-testid="stFileUploaderDropzone"] button {
    border-radius: 8px;
}

/* ---------- Buttons ---------- */
.stButton > button,
.stDownloadButton > button {
    border-radius: 12px;
    font-weight: 600;
    font-size: 1rem;
    padding: 0.7rem 1.5rem;
    transition: all 0.2s ease;
}
/* Primary: Summarize */
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    width: 100%;
    border: none;
    color: #ffffff;
    font-weight: 700;
    font-size: 1.05rem;
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35);
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-2px);
    background: linear-gradient(135deg, #4338ca 0%, #4f46e5 100%);
    box-shadow: 0 8px 20px rgba(79, 70, 229, 0.4);
    color: #ffffff;
}
.stButton > button[kind="primary"]:active,
.stButton > button[data-testid="stBaseButton-primary"]:active {
    transform: translateY(0);
}
/* Secondary: Download */
.stDownloadButton > button {
    background: #ffffff;
    color: var(--accent);
    border: 1.5px solid #c7d2fe;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
}
.stDownloadButton > button:hover {
    background: var(--accent-soft);
    border-color: var(--accent);
    color: var(--accent-dark);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.15);
}

/* ---------- Progress bar ---------- */
[data-testid="stProgress"] > div > div > div { background-color: #e0e7ff; border-radius: 999px; }
[data-testid="stProgress"] > div > div > div > div {
    background: linear-gradient(90deg, #4f46e5, #0ea5e9);
    border-radius: 999px;
}
[data-testid="stProgress"] p { color: var(--muted); font-size: 0.9rem; }

/* ---------- Expander ---------- */
[data-testid="stExpander"] {
    background: var(--card);
    border: 1px solid var(--border) !important;
    border-radius: 14px;
    box-shadow: var(--shadow);
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    padding: 0.8rem 1.1rem;
    font-weight: 600;
    transition: background 0.2s ease;
}
[data-testid="stExpander"] summary:hover { background: var(--accent-soft); }
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    padding: 0.5rem 1.25rem 1.25rem 1.25rem;
    font-size: 0.95rem;
    line-height: 1.7;
}

/* ---------- Result cards (columns) ---------- */
[data-testid="stHorizontalBlock"] { gap: 1.5rem; }
[data-testid="stColumn"] {
    background: var(--card);
    backdrop-filter: blur(8px);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem 1.5rem 1.25rem 1.5rem;
    box-shadow: var(--shadow);
    transition: box-shadow 0.25s ease, transform 0.25s ease;
}
[data-testid="stColumn"]:hover {
    box-shadow: var(--shadow-hover);
    transform: translateY(-2px);
}
[data-testid="stColumn"] h3 {
    font-size: 1.3rem !important;
    margin-bottom: 0.25rem;
    padding-top: 0;
}

/* Word-count badge (captions inside cards) */
[data-testid="stColumn"] [data-testid="stCaptionContainer"] {
    display: inline-block;
    background: var(--accent-soft);
    border-radius: 999px;
    padding: 0.2rem 0.8rem;
    margin-bottom: 0.9rem;
}
[data-testid="stColumn"] [data-testid="stCaptionContainer"] p {
    color: var(--accent-dark) !important;
    font-weight: 600;
    font-size: 0.8rem !important;
    line-height: 1.5;
    margin: 0;
}

/* Original card: scrollable text */
[data-testid="stColumn"]:first-child [data-testid="stMarkdownContainer"] > p {
    max-height: 380px;
    overflow-y: auto;
    padding-right: 0.5rem;
    color: #334155;
    font-size: 0.95rem;
}

/* Summary card: highlighted as the main result */
[data-testid="stColumn"]:nth-child(2) {
    background: linear-gradient(180deg, var(--teal-soft) 0%, #ffffff 100%);
    border: 1px solid var(--teal-border);
    box-shadow: 0 1px 3px rgba(13, 148, 136, 0.08), 0 8px 24px rgba(13, 148, 136, 0.08);
}
[data-testid="stColumn"]:nth-child(2) [data-testid="stCaptionContainer"] {
    background: #ccfbf1;
}
[data-testid="stColumn"]:nth-child(2) [data-testid="stCaptionContainer"] p {
    color: #0f766e !important;
}

/* ---------- Alerts ---------- */
[data-testid="stAlert"] {
    border-radius: 12px;
    border: 1px solid transparent;
    padding: 0.4rem 0.5rem;
}
[data-testid="stAlert"] p { font-size: 1rem; line-height: 1.75; }
/* Success (summary text): soft, readable */
[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) {
    background: #ffffff;
    border: 1px solid var(--teal-border);
    border-left: 4px solid #14b8a6;
    color: var(--text);
}
[data-testid="stAlertContentSuccess"] { color: var(--text); }
[data-testid="stAlertContentSuccess"] p { color: var(--text); }
/* Warning */
[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 4px solid #f59e0b;
}
[data-testid="stAlertContentWarning"] p { color: #92400e; }

/* ---------- Scrollbars ---------- */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

/* ---------- Responsive ---------- */
@media (max-width: 768px) {
    h1 { font-size: 2rem !important; }
    .block-container { padding-top: 1.5rem; }
    [data-testid="stColumn"] { padding: 1.1rem; }
}
@media (prefers-reduced-motion: reduce) {
    * { transition: none !important; }
}
</style>
""", unsafe_allow_html=True)

MODELS = {
    "DistilBART (faster, lighter)": "sshleifer/distilbart-cnn-12-6",
    "BART Large (better quality)": "facebook/bart-large-cnn",
}

LENGTHS = {  # (min_tokens, max_tokens) per chunk
    "Short": (30, 80),
    "Medium": (60, 150),
    "Long": (100, 250),
}

CHUNK_WORDS = 600  # BART accepts ~1024 tokens; 600 words keeps us safely under


@st.cache_resource(show_spinner="Loading model (first run downloads it)...")
def load_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return tokenizer, model


def chunk_text(text: str, size: int = CHUNK_WORDS):
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def summarize_chunk(text, tokenizer, model, min_len, max_len):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    ids = model.generate(
        **inputs,
        min_length=min_len,
        max_length=max_len,
        num_beams=4,
        length_penalty=2.0,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    return tokenizer.decode(ids[0], skip_special_tokens=True)


def summarize(text, tokenizer, model, min_len, max_len):
    chunks = chunk_text(text)
    progress = st.progress(0.0, text="Summarizing...")
    parts = []
    for i, chunk in enumerate(chunks, 1):
        # Very short chunks can't hit min_len; relax it
        words = len(chunk.split())
        parts.append(summarize_chunk(chunk, tokenizer, model, min(min_len, words // 2), max_len))
        progress.progress(i / len(chunks), text=f"Summarized chunk {i}/{len(chunks)}")
    progress.empty()

    combined = " ".join(parts)
    # If the merged summary is still long, summarize it once more
    if len(chunks) > 1 and len(combined.split()) > CHUNK_WORDS:
        combined = summarize_chunk(combined, tokenizer, model, min_len, max_len * 2)
    return combined


def read_pdf(file):
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox("Model", list(MODELS.keys()))
length = st.sidebar.radio("Summary length", list(LENGTHS.keys()), index=1)

# ---------------- Main ----------------
st.title("📝 Text Summarizer")
st.write("Paste text or upload a file (.txt / .pdf) to get a concise summary.")

source = st.radio("Input type", ["Paste text", "Upload file"], horizontal=True)
text = ""
if source == "Paste text":
    text = st.text_area("Your text", height=300, placeholder="Paste an article, notes, or report here...")
else:
    file = st.file_uploader("Upload .txt or .pdf", type=["txt", "pdf"])
    if file:
        text = read_pdf(file) if file.name.lower().endswith(".pdf") else file.read().decode("utf-8", errors="ignore")
        with st.expander("Preview extracted text"):
            st.write(text[:3000] + ("..." if len(text) > 3000 else ""))

if st.button("⚡ Summarize", type="primary"):
    word_count = len(text.split())
    if word_count < 50:
        st.warning("Please provide at least 50 words of text.")
    else:
        tokenizer, model = load_model(MODELS[model_name])
        min_len, max_len = LENGTHS[length]
        summary = summarize(text, tokenizer, model, min_len, max_len)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.caption(f"{word_count} words")
            st.write(text[:5000] + ("..." if len(text) > 5000 else ""))
        with col2:
            st.subheader("Summary")
            s_count = len(summary.split())
            st.caption(f"{s_count} words • {100 - s_count * 100 // word_count}% shorter")
            st.success(summary)
            st.download_button("⬇️ Download summary", summary, "summary.txt", "text/plain")