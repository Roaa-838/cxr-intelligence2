"""
app/app.py
===========
Gradio demo for the Multi-Modal Chest X-Ray Intelligence System.

Two tabs:
  Tab 1 — Report Generation: Upload CXR → model → structured report
  Tab 2 — Clinical Q&A (RAG): Upload CXR + question → retrieved context + answer

Supports:
  - MedGemma-4B-it (zero-shot)
  - CLIP-RAG + MedGemma
  - ColPali-RAG + MedGemma
  - Mock mode (no GPU — instant demo)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import gradio as gr
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MOCK_MODE = os.environ.get("MOCK_MODELS", "true").lower() in ("true", "1", "yes")

# ── Global pipeline instances (lazy-loaded per tab) ──────────────────────────

_report_pipelines: dict = {}
_qa_pipelines: dict     = {}


def _get_report_pipeline(model_name: str):
    """Lazy-load report generation pipeline."""
    key = f"report_{model_name}"
    if key not in _report_pipelines:
        from src.pipelines.report_generation import ReportGenerationPipeline
        _report_pipelines[key] = ReportGenerationPipeline(
            model=model_name, mock=MOCK_MODE
        ).load()
    return _report_pipelines[key]


def _get_qa_pipeline(retriever: str):
    """Lazy-load QA pipeline."""
    key = f"qa_{retriever}"
    if key not in _qa_pipelines:
        from src.pipelines.rag_qa import RAGQAPipeline
        _qa_pipelines[key] = RAGQAPipeline(
            retriever=retriever, generator="medgemma", mock=MOCK_MODE
        ).load()
    return _qa_pipelines[key]


# ── Report Generation handlers ────────────────────────────────────────────────

def generate_report(
    image: Image.Image | None,
    model_choice: str,
    prompt_style: str,
) -> tuple[str, str, str]:
    """
    Handle report generation tab.

    Returns:
        (findings_text, impression_text, metadata_json)
    """
    if image is None:
        return ("⚠️ Please upload a chest X-ray image.", "", "")

    prompt_map = {
        "Standard":  "medgemma_report_v1",
        "Detailed":  "medgemma_report_v2_detailed",
        "RAG-augmented": "medgemma_report_rag",
    }
    prompt_name = prompt_map.get(prompt_style, "medgemma_report_v1")

    model_map = {
        "MedGemma-4B (zero-shot)": "medgemma",
        "MedGemma-4B (fine-tuned)": "medgemma-ft",
        "BiomedCLIP + MedGemma (RAG)": "clip-rag",
        "Real Report Retrieval (CLIP)": "clip-retrieval",
    }
    model_key = model_map.get(model_choice, "medgemma")
    if MOCK_MODE:
        model_key = "mock"

    try:
        pipeline = _get_report_pipeline(model_key)
        result = pipeline.run(image, prompt_template=prompt_name)

        findings   = result.get("findings",   "").strip()
        impression = result.get("impression", "").strip()
        meta = {
            "model":     result.get("model", model_key),
            "latency_s": result.get("latency_s", 0),
            "mock_mode": MOCK_MODE,
            "prompt":    prompt_name,
        }
        if not findings and not impression:
            raw = result.get("raw", "")
            findings   = raw
            impression = ""

        return findings, impression, json.dumps(meta, indent=2)

    except Exception as e:
        logger.exception("Report generation error")
        return f"❌ Error: {e}", "", ""


# ── QA handlers ───────────────────────────────────────────────────────────────

def answer_question(
    image: Image.Image | None,
    question: str,
    retriever_choice: str,
    top_k: int,
    show_context: bool,
) -> tuple[str, str, str]:
    """
    Handle clinical QA tab.

    Returns:
        (answer_text, context_text, retrieved_ids_text)
    """
    if image is None:
        return ("⚠️ Please upload a chest X-ray image.", "", "")
    if not question.strip():
        return ("⚠️ Please enter a clinical question.", "", "")

    retriever_map = {
        "ColPali (visual patch retrieval)":    "colpali",
        "BiomedCLIP (image-text retrieval)":   "clip",
    }
    retriever_key = retriever_map.get(retriever_choice, "clip")
    if MOCK_MODE:
        retriever_key = "clip"  # mock supports any key

    try:
        pipeline = _get_qa_pipeline(retriever_key)
        result = pipeline.run(
            image=image,
            question=question,
            top_k=top_k,
            return_context=show_context,
        )

        answer   = result.get("answer", "")
        context  = result.get("context", "") if show_context else ""
        ids      = "\n".join(result.get("retrieved_studies", []))
        scores   = result.get("retrieval_scores", [])
        scores_s = ", ".join(f"{s:.3f}" for s in scores)

        retrieved_info = f"Study IDs: {ids}\nScores:    {scores_s}"
        return answer, context, retrieved_info

    except Exception as e:
        logger.exception("QA error")
        return f"❌ Error: {e}", "", ""


# ── Example CXR images ────────────────────────────────────────────────────────

def _create_demo_images() -> list[str]:
    """Return paths to real CXR images from the dataset for demo examples."""
    import pandas as pd
    csv_path = Path(__file__).parent.parent / "data" / "processed" / "metadata.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        valid = df[df["image_path"].notna() & (df["split"] == "validate")]
        if len(valid) >= 3:
            return [str(p) for p in valid.iloc[:3]["image_path"]]
    # Fallback: use raw data
    raw_dir = Path(__file__).parent.parent / "data" / "mimic-cxr-raw" / "official_data_iccv_final" / "files" / "p10"
    if raw_dir.exists():
        jpgs = sorted(raw_dir.rglob("*.jpg"))
        if len(jpgs) >= 3:
            return [str(p) for p in jpgs[:3]]
    # Last resort: generate synthetic
    import numpy as np
    sample_dir = Path("app/assets/sample_images")
    sample_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(3):
        p = sample_dir / f"demo_{i}.jpg"
        arr = np.random.randint(30, 200, (448, 448), dtype=np.uint8)
        Image.fromarray(arr).convert("RGB").save(str(p))
        paths.append(str(p))
    return paths


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    demo_images = _create_demo_images()
    mock_badge = "🟡 MOCK MODE (no GPU)" if MOCK_MODE else "🟢 GPU Mode"

    with gr.Blocks(
        title="CXR Intelligence System",
    ) as demo:

        # ── Header ────────────────────────────────────────────────────────────
        gr.HTML(f"""
        <div class="header">
          <h1>🫁 Multi-Modal Chest X-Ray Intelligence System</h1>
          <p>Dual-mode: <b>Report Generation</b> and <b>RAG-Based Clinical Q&A</b></p>
          <p class="badge">Models: MedGemma-4B-it · ColPali-v1.2 · BiomedCLIP &nbsp;|&nbsp; {mock_badge}</p>
        </div>
        """)

        with gr.Tabs():

            # ─── TAB 1: REPORT GENERATION ─────────────────────────────────────
            with gr.TabItem("📋 Report Generation"):
                gr.Markdown(
                    "Upload a chest X-ray and select a model to generate a "
                    "structured radiology report with **Findings** and **Impression** sections."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        img_rg = gr.Image(
                            type="pil",
                            label="Chest X-Ray Image",
                            height=380,
                        )
                        model_rg = gr.Dropdown(
                            choices=[
                                "Real Report Retrieval (CLIP)",
                                "MedGemma-4B (zero-shot)",
                                "MedGemma-4B (fine-tuned)",
                                "BiomedCLIP + MedGemma (RAG)",
                            ],
                            value="Real Report Retrieval (CLIP)",
                            label="Model",
                        )
                        prompt_style = gr.Dropdown(
                            choices=["Standard", "Detailed", "RAG-augmented"],
                            value="Standard",
                            label="Prompt Style",
                        )
                        gr.Examples(
                            examples=[[str(p)] for p in demo_images],
                            inputs=[img_rg],
                            label="Demo Images (synthetic — replace with real CXR)",
                        )
                        gen_btn = gr.Button(
                            "🔬 Generate Report", variant="primary", size="lg"
                        )

                    with gr.Column(scale=2):
                        findings_out = gr.Textbox(
                            label="FINDINGS",
                            lines=10,
                            placeholder="Systematic description of all visible structures …",
                        )
                        impression_out = gr.Textbox(
                            label="IMPRESSION",
                            lines=4,
                            placeholder="Clinical summary and key conclusions …",
                        )
                        meta_out = gr.JSON(label="Metadata", visible=False)
                        show_meta = gr.Checkbox(
                            label="Show metadata",
                            value=False,
                        )
                        show_meta.change(
                            lambda show: gr.update(visible=show),
                            inputs=show_meta,
                            outputs=meta_out,
                        )

                gen_btn.click(
                    generate_report,
                    inputs=[img_rg, model_rg, prompt_style],
                    outputs=[findings_out, impression_out, meta_out],
                )

            # ─── TAB 2: RAG QA ────────────────────────────────────────────────
            with gr.TabItem("❓ Clinical Q&A (RAG)"):
                gr.Markdown(
                    "Ask a clinical question about a chest X-ray. The system **retrieves "
                    "similar training cases** from the knowledge base and uses them as "
                    "context for the answer."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        img_qa = gr.Image(
                            type="pil",
                            label="Chest X-Ray Image",
                            height=280,
                        )
                        question_in = gr.Textbox(
                            label="Clinical Question",
                            placeholder=(
                                "Is there a pleural effusion?\n"
                                "Where is the consolidation located?\n"
                                "Is this chest X-ray normal?"
                            ),
                            lines=3,
                        )
                        retriever_dd = gr.Dropdown(
                            choices=[
                                "ColPali (visual patch retrieval)",
                                "BiomedCLIP (image-text retrieval)",
                            ],
                            value="ColPali (visual patch retrieval)",
                            label="Retrieval Model",
                        )
                        top_k_sl = gr.Slider(
                            minimum=1, maximum=10, step=1, value=5,
                            label="Retrieved Cases (K)",
                        )
                        show_ctx = gr.Checkbox(
                            label="Show retrieved context",
                            value=False,
                        )
                        gr.Examples(
                            examples=[
                                ["Is there a pleural effusion in this chest X-ray?"],
                                ["Is cardiomegaly present?"],
                                ["What abnormalities are visible?"],
                                ["Is this chest X-ray within normal limits?"],
                                ["Where is the consolidation located?"],
                            ],
                            inputs=[question_in],
                            label="Example Questions",
                        )
                        qa_btn = gr.Button(
                            "🔍 Answer Question", variant="primary", size="lg"
                        )

                    with gr.Column(scale=2):
                        answer_out = gr.Textbox(
                            label="Answer",
                            lines=6,
                            placeholder="RAG-grounded answer will appear here …",
                        )
                        context_out = gr.Textbox(
                            label="Retrieved Context (from training knowledge base)",
                            lines=10,
                            visible=False,
                            placeholder="Retrieved reports from similar cases …",
                        )
                        retrieved_out = gr.Textbox(
                            label="Retrieved Study IDs & Scores",
                            lines=3,
                        )

                show_ctx.change(
                    lambda v: gr.update(visible=v),
                    inputs=show_ctx,
                    outputs=context_out,
                )
                qa_btn.click(
                    answer_question,
                    inputs=[img_qa, question_in, retriever_dd, top_k_sl, show_ctx],
                    outputs=[answer_out, context_out, retrieved_out],
                )

            # ─── TAB 3: MODEL COMPARISON ──────────────────────────────────────
            with gr.TabItem("📊 Model Comparison"):
                gr.Markdown("""
## Model Comparison Overview

| Model | Role | Strengths | Limitations |
|-------|------|-----------|-------------|
| **MedGemma-4B-it** | Report generation + QA | Medical-specific training, MedSigLIP vision encoder, open weights | Requires ~9GB VRAM, gated model |
| **ColPali-v1.2** | Visual RAG retrieval | Preserves spatial layout, patch-level attention | Not domain-specific (CXR), heavy indexing |
| **BiomedCLIP** | Image-text retrieval | Medical domain trained, faster than ColPali | Single-vector (loses spatial info) |
| **CLIP ViT-L/14** | Baseline retrieval | Fast, widely adopted | General domain only |

## Evaluation Metrics

| Metric | What it measures | Clinical validity |
|--------|-----------------|-------------------|
| **RadGraph F1** ⭐ | Clinical entity + relation graph overlap | Highest (validated by radiologists) |
| **BERTScore F1** | Semantic embedding similarity | Medium-High |
| **ROUGE-L** | Longest common subsequence | Medium |
| **BLEU-4** | N-gram precision | Low (doesn't capture clinical meaning) |
| **Token F1 (QA)** | Answer token overlap | Medium |
| **Exact Match (QA)** | Yes/No accuracy | High for presence questions |

## Architecture Diagram

```
Mode 1 — Report Generation:
  CXR Image → [MedGemma MedSigLIP encoder] → [Gemma 3 4B LLM] → Report
  CXR Image → [BiomedCLIP embed] → [Top-K similar reports] → [MedGemma] → Report

Mode 2 — RAG-Based QA:
  (Image + Question) → [ColPali: 1024 patch embeds × 128d] ─┐
                                                             ├→ MaxSim → Top-K
  Training CXR Index ─────────────────────────────────────── ┘
  Top-K reports → [MedGemma] → Grounded answer
```

## QA Dataset Creation Methodology

See `docs/qa_dataset_creation.md` for full write-up. Summary:

1. Used the **test split** reports (ground truth text) as source material
2. Prompted **Claude-3.5-Haiku** to generate 5 QA pairs per report
3. Covered 5 question types: presence, location, severity, diagnosis, comparison
4. Applied quality filters: min length, deduplication, degenerate answer removal
5. **Resulted in ~2,500 QA pairs from 500 test studies**
6. Validated 100 random samples manually against source reports
                """)

        # ── Footer ────────────────────────────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center; margin-top:20px; font-size:0.8em; color:#999;">
          Models: MedGemma-4B-it (Google) · ColPali-v1.2 (Vidore) · BiomedCLIP (Microsoft) |
          Dataset: MIMIC-CXR (PhysioNet / Kaggle) |
          <a href="https://github.com/your-username/cxr-intelligence" target="_blank">GitHub</a>
        </div>
        """)

    return demo


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--share",   action="store_true", help="Create public share link")
    p.add_argument("--port",    type=int, default=7860)
    p.add_argument("--host",    default="0.0.0.0")
    p.add_argument("--no-mock", action="store_true", help="Load real models (requires GPU)")
    args = p.parse_args()

    global MOCK_MODE
    if args.no_mock:
        MOCK_MODE = False
        os.environ["MOCK_MODELS"] = "false"

    logger.info("Starting CXR Intelligence System (mock=%s) …", MOCK_MODE)
    app = build_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="cyan",
        ),
        css="""
        .header { text-align: center; padding: 10px; }
        .badge  { font-size: 0.85em; color: #666; }
        footer  { display: none !important; }
        """,
    )


if __name__ == "__main__":
    main()
