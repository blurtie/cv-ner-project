"""
CV Screening App — Resume NER Demo (RoBERTa only)
===================================================
Streamlit app for the "Automated CV Screening with NER" project.

Uses Model 4 (RoBERTa) only — it scored the highest F1 (0.525) among the
4 models compared in this project, so it's the one used in production.

Run with:  streamlit run app.py
"""

import html
from collections import defaultdict
from pathlib import Path

import streamlit as st

# ──────────────────────────────────────────────────────────────────
# CONFIG — local folder path, OR a Hugging Face Hub repo id
# (e.g. "username-kamu/roberta-cv-ner") if you uploaded the model there
# instead of committing it to git.
# ──────────────────────────────────────────────────────────────────
ROBERTA_PATH = "blurtie/roberta-cv-ner"

ENTITY_LABELS = [
    "Name", "Email Address", "Skills", "Degree", "College Name",
    "Location", "Companies worked at", "Designation",
    "Graduation Year", "Years of Experience",
]

# Distinct, readable background color per entity type
ENTITY_COLORS = {
    "Name":                 "#FFADAD",
    "Email Address":        "#FFC6FF",
    "Skills":               "#A0C4FF",
    "Degree":               "#FDFFB6",
    "College Name":         "#FFD6A5",
    "Location":             "#BDB2FF",
    "Companies worked at":  "#CAFFBF",
    "Designation":          "#9BF6FF",
    "Graduation Year":      "#D9D9D9",
    "Years of Experience":  "#FFB4A2",
}


# ──────────────────────────────────────────────────────────────────
# Model loading — RoBERTa via HuggingFace Transformers
# ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model RoBERTa...")
def load_roberta_pipeline(path: str):
    from transformers import (AutoModelForTokenClassification, AutoTokenizer, pipeline)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForTokenClassification.from_pretrained(path)
    # RoBERTa doesn't use token_type_ids — some tokenizer configs still
    # emit them by default, which crashes forward(). Strip it defensively.
    if "token_type_ids" in tokenizer.model_input_names:
        tokenizer.model_input_names = [
            n for n in tokenizer.model_input_names if n != "token_type_ids"
        ]
    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")


def extract_entities(text: str):
    """Run RoBERTa NER → list of (start, end, label, score).

    ROBERTA_PATH can be a local folder OR a Hugging Face Hub repo id
    (e.g. "username/roberta-cv-ner") — from_pretrained() handles both
    transparently, so we just try loading and report a clear error if
    neither resolves.
    """
    try:
        ner_pipe = load_roberta_pipeline(ROBERTA_PATH)
    except OSError as err:
        raise FileNotFoundError(
            f"Tidak bisa memuat model dari '{ROBERTA_PATH}'. Cek salah satu:\n"
            "1) Kalau ini folder lokal — pastikan kamu sudah download hasil "
            "training RoBERTa dari Colab (Notebook 04) dan taruh di lokasi ini.\n"
            "2) Kalau ini repo id Hugging Face Hub — pastikan repo-nya ada "
            "dan public (atau sudah login via `huggingface-cli login` kalau "
            "private).\n"
            f"Pesan asli: {err}"
        ) from err

    results = ner_pipe(text)
    return [(int(r["start"]), int(r["end"]), r["entity_group"], float(r["score"]))
            for r in results]


# ──────────────────────────────────────────────────────────────────
# Build highlighted HTML from raw text + entity spans
# ──────────────────────────────────────────────────────────────────
def highlight_text(text: str, entities):
    entities = sorted(entities, key=lambda e: e[0])
    clean = []
    last_end = -1
    for s, e, label, score in entities:
        if s < last_end:
            continue
        clean.append((s, e, label, score))
        last_end = e

    out = []
    cursor = 0
    for s, e, label, score in clean:
        out.append(html.escape(text[cursor:s]))
        color = ENTITY_COLORS.get(label, "#E0E0E0")
        span_text = html.escape(text[s:e])
        tooltip = f"{label}" + (f" ({score:.2f})" if score is not None else "")
        out.append(
            f'<mark style="background-color:{color}; padding:1px 3px; '
            f'border-radius:4px; margin:0 1px;" title="{tooltip}">'
            f'{span_text}'
            f'<span style="font-size:0.65em; font-weight:600; '
            f'vertical-align:super; margin-left:2px;">{html.escape(label)}</span>'
            f'</mark>'
        )
        cursor = e
    out.append(html.escape(text[cursor:]))
    return "".join(out).replace("\n", "<br>")


# ──────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="CV Screening — NER Demo", layout="wide")

st.title("📄 Automated CV Screening — NER Demo")
st.caption(
    "Ekstraksi otomatis informasi penting dari CV (Skills, Pendidikan, "
    "Pengalaman, dll.) menggunakan model **RoBERTa** — model dengan "
    "F1-Score tertinggi (0.525) di antara 4 model yang diuji pada proyek ini."
)

with st.sidebar:
    st.header("ℹ️ Model")
    st.success("**Model 4 — RoBERTa**\n\nF1-Score: 0.525 (tertinggi)")
    st.caption(
        "Model lain (Rule-Based, CNN, DistilBERT) dipakai untuk perbandingan "
        "di laporan, tapi RoBERTa yang dipakai di aplikasi ini karena "
        "performanya paling baik."
    )

    st.divider()
    st.subheader("Legenda Entitas")
    for label in ENTITY_LABELS:
        color = ENTITY_COLORS[label]
        st.markdown(
            f'<div style="display:flex; align-items:center; margin-bottom:4px;">'
            f'<div style="width:14px; height:14px; background:{color}; '
            f'border-radius:3px; margin-right:8px;"></div>'
            f'<span style="font-size:0.85em;">{label}</span></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption(
        "💡 Butuh file model hasil training RoBERTa (Notebook 04). "
        "Path bisa diubah di `ROBERTA_PATH` pada app.py."
    )

col_input, col_output = st.columns([1, 1.3], gap="large")

with col_input:
    st.subheader("1️⃣ Input CV")

    source = st.radio("Sumber teks", ["Upload .txt", "Paste manual"],
                       horizontal=True)

    if source == "Upload .txt":
        uploaded = st.file_uploader("Upload file CV (.txt)", type=["txt"])
        cv_text = uploaded.read().decode("utf-8", errors="ignore") if uploaded else ""
        st.text_area("Preview", value=cv_text, height=420, disabled=True)
    else:
        cv_text = st.text_area("Paste teks CV di sini", height=420,
                                placeholder="Copy-paste teks mentah CV...")

    run = st.button("🔍 Ekstrak Entitas", type="primary", use_container_width=True)

with col_output:
    st.subheader("2️⃣ Hasil Ekstraksi")

    if run:
        if not cv_text.strip():
            st.warning("Teks CV masih kosong — isi dulu di panel kiri.")
        else:
            try:
                with st.spinner("Menjalankan model RoBERTa..."):
                    entities = extract_entities(cv_text)

                if not entities:
                    st.info("Tidak ada entitas yang terdeteksi untuk teks ini.")
                else:
                    tab1, tab2 = st.tabs(["✨ Highlight", "📋 Tabel Entitas"])

                    with tab1:
                        st.markdown(highlight_text(cv_text, entities),
                                    unsafe_allow_html=True)

                    with tab2:
                        grouped = defaultdict(list)
                        for s, e, label, score in entities:
                            grouped[label].append(cv_text[s:e].strip())

                        for label in ENTITY_LABELS:
                            if label in grouped:
                                with st.expander(f"**{label}** ({len(grouped[label])})",
                                                  expanded=True):
                                    for val in grouped[label]:
                                        st.write(f"• {val}")

            except FileNotFoundError as err:
                st.error(str(err))
            except Exception as err:  # noqa: BLE001
                st.error(f"Terjadi error saat menjalankan model: {err}")
    else:
        st.info("Klik **Ekstrak Entitas** untuk mulai memproses CV di panel kiri.")
