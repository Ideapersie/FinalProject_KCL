"""
ui/app.py — Gradio Blocks front end for the adaptive-gating demo.

Layout A from the design: one scrolling column — settings accordion, question,
verdict banner, entropy heatmap, P5-vs-P3 answers, evidence. All rendering
logic lives in attribution.py and all orchestration in session.py; this file
only wires widgets to those, so it needs no tests of its own.
"""

from __future__ import annotations

import html

import gradio as gr

from medrag_adaptive.models.openai_backend import OpenAIBackend
from medrag_adaptive.ui.attribution import (
    align_draft,
    render_chunks_html,
    render_entropy_html,
    truncated_entropy,
)
from medrag_adaptive.ui.session import DemoResult, DemoSession

CSS = """
.mr-heat { line-height: 2.1; }
.mr-tok { padding: 2px 4px; margin: 1px; border-radius: 3px;
          font-family: ui-monospace, monospace; font-size: 13px; }
.mr-note { font-size: 12px; opacity: .75; margin-top: 8px; font-style: italic; }
.mr-term { background: #ffe27a; color: #111; border-radius: 2px; }
.mr-chunk { border: 1px solid rgba(128,128,128,.35); border-radius: 6px;
            padding: 10px; margin-bottom: 8px; }
.mr-chunk-head { font-weight: 600; margin-bottom: 6px; }
.mr-chunk-body { font-size: 13px; line-height: 1.5; }
.mr-badge { display: inline-block; border: 1px solid rgba(128,128,128,.4);
            border-radius: 10px; padding: 0 7px; font-size: 11px;
            font-weight: 400; margin-left: 6px; }
.mr-verdict { padding: 10px 14px; border-radius: 6px; font-weight: 600;
              color: #fff; font-size: 15px; }
.mr-retrieve { background: #c0392b; }
.mr-skip { background: #27795b; }
"""


def _verdict_html(result: DemoResult) -> str:
    css_class = "mr-retrieve" if result.verdict == "RETRIEVE" else "mr-skip"
    signal = "n/a" if result.signal_value is None else f"{result.signal_value:.3f}"
    threshold = "n/a" if result.threshold is None else f"{result.threshold:.3f}"

    parts = [
        f'<div class="mr-verdict {css_class}">{result.verdict}'
        f" &nbsp;·&nbsp; {html.escape(result.gate_name)} gate"
        f" &nbsp;·&nbsp; signal {signal} vs τ {threshold}"
        f" &nbsp;·&nbsp; {result.latency_s:.1f}s</div>"
    ]

    votes = (result.gate_details or {}).get("votes")
    if votes:
        rendered = ", ".join(f"{name}: {vote}" for name, vote in votes.items())
        need = result.gate_details.get("min_votes")
        parts.append(
            f'<div class="mr-note">Member votes — {html.escape(rendered)} '
            f"(needs {need} to retrieve)</div>"
        )
    for note in result.notes:
        parts.append(f'<div class="mr-note">{html.escape(note)}</div>')
    return "".join(parts)


def _error_html(message: str) -> str:
    return ('<div class="mr-verdict mr-retrieve">Could not run</div>'
            f'<div class="mr-note">{html.escape(message)}</div>')


def build_app(session: DemoSession) -> "gr.Blocks":
    # NOTE: css is NOT passed here. Gradio 6 moved it to launch(), and passing
    # it to the constructor is only a warning — the styling is silently dropped,
    # which would leave every panel unstyled. launch_app() below applies it.
    with gr.Blocks(title="Adaptive Gating for Medical RAG") as demo:
        gr.Markdown(
            "## Adaptive gating for medical RAG\n"
            "Ask a clinical question. The gate decides whether the model needs to "
            "retrieve evidence, and the panels below show why it decided that."
        )

        with gr.Accordion("Settings", open=False):
            with gr.Row():
                gate_type = gr.Dropdown(
                    ["ensemble", "entropy", "margin", "hallucination_probe"],
                    value=session.cfg.gate.type,
                    label="Gate",
                )
                top_k = gr.Slider(1, 10, value=session.cfg.retrieval.top_k,
                                  step=1, label="top-k chunks")
            with gr.Row():
                tau_h = gr.Slider(0.0, 3.0, value=session.cfg.gate.entropy_threshold,
                                  step=0.01, label="τ entropy (retrieve when H̄ >)")
                tau_m = gr.Slider(0.0, 1.0, value=session.cfg.gate.margin_threshold,
                                  step=0.01, label="τ margin (retrieve when M̄ <)")
            gr.Markdown(
                "**Backend** — the local GGUF is loaded. An OpenAI-compatible API "
                "can answer and run the margin and probe gates, but cannot supply "
                "full-vocabulary logits, so the entropy gate abstains there."
            )
            with gr.Row():
                api_base = gr.Textbox(label="base_url",
                                      placeholder="https://api.openai.com/v1")
                api_model = gr.Textbox(label="model", placeholder="gpt-4o-mini")
                api_key = gr.Textbox(label="API key", type="password")
            with gr.Row():
                use_api = gr.Button("Use API backend")
                use_local = gr.Button("Use local GGUF")
            backend_status = gr.Markdown(
                "Backend: **local GGUF**" if session.local_llm is not None
                else "Backend: **none loaded** — configure an API above."
            )

        question = gr.Textbox(
            label="Question", lines=3,
            placeholder="A patient develops anaphylactic shock after a bee sting. "
                        "What is the first-line drug?",
        )
        with gr.Row():
            choice_a = gr.Textbox(label="A", scale=1)
            choice_b = gr.Textbox(label="B", scale=1)
            choice_c = gr.Textbox(label="C", scale=1)
            choice_d = gr.Textbox(label="D", scale=1)
        gr.Markdown("*Leave the choices blank to ask a free-text question.*")
        submit = gr.Button("Run", variant="primary")

        verdict_out = gr.HTML()
        gr.Markdown("### Draft-token entropy")
        gr.Markdown(
            "Each chip is one token of the model's unretrieved draft, shaded by "
            "H(p_t). Darker means the model had less idea what came next. Hover "
            "for the value in nats.\n\n"
            "*The draft's first token is not shown: the entropy gate's logit "
            "slice begins one row late, so its H̄ — the value in the banner, and "
            "the one every reported result was computed from — averages tokens 2 "
            "onward. The chips below are exactly the tokens that average covers.*"
        )
        heatmap_out = gr.HTML()
        with gr.Row():
            p5_out = gr.Textbox(label="P5 — gated", lines=6, interactive=False)
            p3_out = gr.Textbox(label="P3 — closed book", lines=6, interactive=False)
        identical_out = gr.Markdown()
        gr.Markdown("### Evidence")
        chunks_out = gr.HTML()

        # ── callbacks ──────────────────────────────────────────────

        def _run(q, a, b, c, d, gate, th, tm, k):
            choices = {letter: text.strip() for letter, text in
                       (("A", a), ("B", b), ("C", c), ("D", d))
                       if text and text.strip()}
            try:
                result = session.answer(
                    q, choices=choices or None, gate_type=gate,
                    entropy_threshold=th, margin_threshold=tm, top_k=int(k),
                )
            except Exception as exc:              # surfaced in the UI, not swallowed
                return _error_html(str(exc)), "", "", "", "", ""

            aligned = result.aligned
            caveat = ""
            if not aligned.tokens and isinstance(session.llm, OpenAIBackend):
                topk = truncated_entropy(session.llm.last_draft_topk)
                if topk:
                    aligned = align_draft(
                        [f"t{i}" for i in range(len(topk.per_token))], topk.per_token
                    )
                    caveat = ("Entropy computed from the API's top-k logprobs — a "
                              "truncated lower bound, not comparable to the "
                              "calibrated τ, and the token labels are positions "
                              "rather than the model's own token strings.")

            heat = render_entropy_html(aligned, threshold=result.threshold or 0.0)
            if caveat:
                heat += f'<div class="mr-note">{html.escape(caveat)}</div>'

            identical = ""
            if result.answers_identical:
                identical = ("*The gate skipped retrieval, so both columns ran the "
                             "same closed-book prompt and are identical by "
                             "construction — this is not a comparison.*")

            chunks_html = (
                render_chunks_html(
                    result.chunks, result.query,
                    lexical_only=session.cfg.policy.retrieval_mode != "bm25",
                )
                if result.chunks
                else '<div class="mr-note">The gate skipped retrieval, so no '
                     "evidence was fetched.</div>"
            )
            return (_verdict_html(result), heat, result.p5_answer,
                    result.p3_answer, identical, chunks_html)

        def _switch_api(base, model, key):
            if not base or not model:
                return "Backend unchanged — base_url and model are both required."
            session.llm = OpenAIBackend(base_url=base, model=model, api_key=key or None)
            return f"Backend: **API** — `{model}` at `{base}`"

        def _switch_local():
            if session.local_llm is None:
                return "No local GGUF was loaded at startup."
            session.llm = session.local_llm
            return "Backend: **local GGUF**"

        submit.click(
            _run,
            [question, choice_a, choice_b, choice_c, choice_d,
             gate_type, tau_h, tau_m, top_k],
            [verdict_out, heatmap_out, p5_out, p3_out, identical_out, chunks_out],
        )
        use_api.click(_switch_api, [api_base, api_model, api_key], backend_status)
        use_local.click(_switch_local, None, backend_status)

    # One Llama instance, and it is not thread-safe: serialise submissions.
    demo.queue(default_concurrency_limit=1)
    return demo


def launch_app(session: DemoSession, port: int = 7860, inbrowser: bool = True) -> None:
    """Build and serve the demo on localhost.

    CSS is applied here rather than in build_app because Gradio 6 accepts it
    only at launch time; keeping the knowledge in this module means the
    launcher script never has to know the app has styling at all.
    """
    build_app(session).launch(
        css=CSS,
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        inbrowser=inbrowser,
    )
