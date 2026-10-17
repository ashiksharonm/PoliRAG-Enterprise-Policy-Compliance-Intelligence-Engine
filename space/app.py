"""Gradio demo: policy change impact analysis. Runs on CPU with MiniLM (or hashing fallback)."""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
if not (ROOT / "src").exists():  # running from the repo checkout: space/ is one level down
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.impact.analysis import (  # noqa: E402
    ImpactConfig,
    analyze_impact,
    corpus_from_texts,
    load_corpus,
    load_questions,
    render_markdown,
)
from src.impact.embedders import Embedder, make_embedder  # noqa: E402

V1, V2 = ROOT / "data" / "raw", ROOT / "data" / "versions" / "v2"
GOLD = ROOT / "data" / "eval" / "retrieval_gold.json"
_embedder: Embedder | None = None


def embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        try:
            _embedder = make_embedder(
                "sentence-transformers", "sentence-transformers/all-MiniLM-L6-v2"
            )
        except Exception:  # model download or torch unavailable: lexical fallback
            _embedder = make_embedder("hashing", "")
    return _embedder


def run(doc_name: str, old_text: str, new_text: str, k: int) -> str:
    """Compare one pasted policy pair against the other bundled documents unchanged."""
    old, new = load_corpus(V1), load_corpus(V1)
    if old_text.strip() or new_text.strip():
        old = corpus_from_texts({**old.raw, doc_name: old_text})
        new = corpus_from_texts({**new.raw, doc_name: new_text})
    else:
        new = load_corpus(V2)
    report = analyze_impact(
        old, new, load_questions(GOLD), embedder(), None, ImpactConfig(k=int(k))
    )
    return render_markdown(report)


def sample_pair() -> tuple[str, str, str]:
    name = "privacy_data_subject_rights.txt"
    return name, (V1 / name).read_text(), (V2 / name).read_text()


def build() -> gr.Blocks:
    with gr.Blocks(title="PoliRAG policy change impact") as demo:
        gr.Markdown(
            "# Policy change impact analysis\nDiff two versions of a policy at clause level, then see "
            "which of the 76 gold questions get different retrieved evidence. Leave both boxes empty "
            "to compare the bundled v1 and v2 corpora (synthetic edits to fictional ACME policies). "
            "Retrieval only, answers are extractive and cite clause ids."
        )
        name = gr.Textbox(
            label="Policy file name to replace", value="privacy_data_subject_rights.txt"
        )
        with gr.Row():
            old = gr.Textbox(label="Old version (paste text)", lines=14)
            new = gr.Textbox(label="New version (paste text)", lines=14)
        k = gr.Slider(1, 3, value=2, step=1, label="Evidence depth k")
        out = gr.Markdown()
        with gr.Row():
            gr.Button("Load sample pair").click(sample_pair, None, [name, old, new])
            gr.Button("Analyze", variant="primary").click(run, [name, old, new, k], out)
    return demo


if __name__ == "__main__":
    build().launch(server_name="0.0.0.0", server_port=7860)
