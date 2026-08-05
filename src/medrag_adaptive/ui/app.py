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
    render_agreement_html,
    render_chunks_html,
    render_entropy_html,
    render_gate_table_html,
    truncated_entropy,
)
from medrag_adaptive.ui.session import DemoResult, DemoSession

# Four MIRAGE-MMLU questions whose verdicts are known from the logged 200-question
# run (results/raw_logs/p5_medcorp_mcq.jsonl), picked because their ensembles were
# unanimous — 3/3 or 0/3 — so they are the least likely to flip on the day. The
# gate is greedy and its draft length is fixed by config, so the demo reproduces
# the logged decision; the recorded signals are in the comments as a sanity check.
#
# The wording is copied verbatim from data/raw/mirage/benchmark.json. It has to
# be: paraphrasing one of these changed its H̄ from 1.53 to 0.96 in testing, which
# is the difference between a preset that demonstrates the gate and one that
# quietly contradicts the logged run on stage.
EXAMPLES = [
    # anatomy-020, 0/3 votes — H̄ 0.631, M̄ 0.800, probe agrees.
    ["Where is the sinoatrial node located?",
     "Between the left atrium and the left ventricle",
     "Between the right atrium and the right ventricle",
     "In the upper wall of the right atrium",
     "In the upper wall of the left ventricle", "C"],
    # anatomy-004, 0/3 votes — H̄ 0.536, M̄ 0.784, probe agrees.
    ["Which of the following describes the cluster of blood capillaries found in "
     "each nephron in the kidney?",
     "Afferent arteriole", "Glomerulus", "Loop of Henle", "Renal pelvis", "B"],
    # anatomy-075, 3/3 votes — H̄ 1.532, M̄ 0.462, probe disagrees.
    ["A patient cuts a peripheral motor nerve in their wrist when they fall through "
     "a plate glass window. If the nerve does not regenerate, after about 6 months "
     "the muscles it normally innervates will show signs of which of the four "
     "options below?",
     "spastic paralysis", "flaccid paralysis", "atrophy", "contracture", "C"],
    # anatomy-024, 3/3 votes — H̄ 1.068, M̄ 0.597, probe disagrees. The 3B gets
    # this one wrong under both policies, which is worth showing rather than hiding.
    ["The spheno-occipital synchondrosis",
     "is a secondary growth cartilage.",
     "influences the position of the viscerocranium.",
     "ceases activity at 7 years of age.",
     "can be reactivated in patients affected by acromegaly.", "B"],
]

# The `.dark` selector is repeated on every variable deliberately: Gradio picks
# dark or light from the viewer's OS setting, and a projector laptop that happens
# to be in dark mode would otherwise invert the whole demo mid-presentation.
# Pinning both keeps one known-good palette on any machine.
CSS = """
:root, .dark {
  --body-background-fill: #f4f5f7;
  --background-fill-primary: #ffffff;
  --background-fill-secondary: #eceef1;
  --block-background-fill: #ffffff;
  --block-label-background-fill: #ffffff;
  --block-label-text-color: #5b636c;
  --block-title-text-color: #5b636c;
  --input-background-fill: #ffffff;
  --input-border-color: #d7dbe0;
  --body-text-color: #1b1f24;
  --body-text-color-subdued: #5b636c;
  --border-color-primary: #d7dbe0;
  --border-color-accent: #c3c9d1;
  --table-odd-background-fill: #f7f8fa;
  --table-even-background-fill: #ffffff;
  --table-text-color: #2b3138;
  --color-accent: #2f5d8a;
  --color-accent-soft: #eef2f6;
  --link-text-color: #2f5d8a;
  --button-primary-background-fill: #2b3138;
  --button-primary-background-fill-hover: #1b1f24;
  --button-primary-text-color: #ffffff;
  --button-primary-border-color: #2b3138;
  --button-secondary-background-fill: #ffffff;
  --button-secondary-text-color: #2b3138;
}
.gradio-container { background: #f4f5f7;
                    font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
/* Gradio's example rows inherit a muted fill that leaves grey text on grey. */
.gradio-container table td, .gradio-container table th { color: #2b3138; }
label > span, .block > label > span { color: #5b636c !important;
                                      background: transparent !important; }
/* The accordion header is a label too, and the rule above leaves it too faint. */
.label-wrap > span, .label-wrap { color: #1b1f24 !important; font-weight: 600; }
/* Hovering or selecting an example row otherwise flips it to the dark-theme
   fill while the text stays dark — the row goes unreadable exactly as it is
   clicked, which is the one moment the audience is looking at it. */
.gradio-container tbody tr:hover td,
.gradio-container tbody tr.selected td,
.gradio-container tbody tr:hover th { background: #eef2f6 !important;
                                      color: #1b1f24 !important; }
.mr-heat { line-height: 2.0; }
.mr-tok { padding: 2px 4px; margin: 1px; border-radius: 3px;
          font-family: ui-monospace, monospace; font-size: 13px; }
.mr-note { font-size: 12px; color: #6b7078; margin-top: 8px; }
.mr-term { background: #ffe27a; color: #111; border-radius: 2px; }
.mr-chunk { border: 1px solid #dfe3e8; border-radius: 6px; background: #fff;
            padding: 10px; margin-bottom: 8px; }
.mr-chunk-head { font-weight: 600; margin-bottom: 6px; color: #1b1f24; }
.mr-chunk-body { font-size: 13px; line-height: 1.5; color: #2b3138; }
.mr-badge { display: inline-block; border: 1px solid #d7dbe0; color: #5b636c;
            border-radius: 10px; padding: 0 7px; font-size: 11px;
            font-weight: 400; margin-left: 6px; }
.mr-verdict { padding: 10px 14px; border-radius: 6px; font-weight: 600;
              color: #fff; font-size: 15px; }
.mr-retrieve { background: #b3452f; }
.mr-skip { background: #2f7d5b; }
.mr-gates { border-collapse: collapse; margin-top: 10px; font-size: 13px;
            background: #fff; }
.mr-gates th, .mr-gates td { border: 1px solid #dfe3e8; padding: 5px 12px;
                             text-align: left; color: #2b3138; }
.mr-gates th { font-weight: 600; color: #5b636c; background: #f7f8fa; }
.mr-vote-retrieve { color: #b3452f; font-weight: 600; }
.mr-vote-skip { color: #2f7d5b; font-weight: 600; }
.mr-agree { display: flex; align-items: center; gap: 8px; padding: 8px 12px;
            border-radius: 6px; font-size: 14px; border: 1px solid; }
.mr-agree-good { background: #e8f5ee; border-color: #86c4a4; color: #1d5c41; }
.mr-agree-split { background: #fdf4e3; border-color: #e0bd77; color: #7a5410; }
.mr-agree-bad { background: #fbecea; border-color: #e0a79c; color: #8a3323; }
.mr-chip { border-radius: 4px; padding: 1px 8px; font-weight: 600;
           font-size: 13px; }
.mr-chip-good { background: #2f7d5b; color: #fff; }
.mr-chip-bad { background: #b3452f; color: #fff; }
.mr-chip-gold { background: #e7eaee; color: #2b3138; }
"""


def _verdict_html(result: DemoResult) -> str:
    css_class = "mr-retrieve" if result.verdict == "RETRIEVE" else "mr-skip"

    # The ensemble's "signal" is a vote count and its "τ" is min_votes. Printing
    # those as 3.000 vs τ 2.000 invites the reader to see a continuous score
    # crossing a calibrated threshold, which is exactly what it is not — so the
    # ensemble gets counted out in votes and only single gates show a τ.
    need = (result.gate_details or {}).get("min_votes")
    if need is not None:
        members = len((result.gate_details or {}).get("votes") or {})
        got = (result.gate_details or {}).get("retrieve_votes")
        summary = f"{got} of {members} members voted retrieve, needs {need}"
    else:
        signal = "n/a" if result.signal_value is None else f"{result.signal_value:.3f}"
        threshold = "n/a" if result.threshold is None else f"{result.threshold:.3f}"
        summary = f"signal {signal} vs τ {threshold}"

    parts = [
        f'<div class="mr-verdict {css_class}">{result.verdict}'
        f" &nbsp;·&nbsp; {html.escape(result.gate_name)} gate"
        f" &nbsp;·&nbsp; {summary}"
        f" &nbsp;·&nbsp; {result.latency_s:.1f}s</div>"
    ]
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
            "The gate decides whether the model needs evidence. The panels show why."
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
        with gr.Row():
            gold = gr.Textbox(label="Correct answer (optional)", scale=1,
                              placeholder="C")
            gr.Markdown("Leave the choices blank for a free-text question. "
                        "Give a letter here to score both policies against it.")
        submit = gr.Button("Run", variant="primary")
        gr.Examples(
            EXAMPLES,
            inputs=[question, choice_a, choice_b, choice_c, choice_d, gold],
            label="Examples — first two skipped, last two retrieved in the logged run",
        )

        status_out = gr.Markdown()
        verdict_out = gr.HTML()
        gate_table_out = gr.HTML()
        gr.Markdown("### Draft-token entropy")
        gr.Markdown(
            "One chip per token of the unretrieved draft, shaded by H(p_t) — "
            "darker is less certain. Hover for nats. The first token is omitted: "
            "the gate's H̄ averages from token 2 onward, and these are exactly the "
            "tokens it averaged."
        )
        heatmap_out = gr.HTML()
        with gr.Row():
            p5_out = gr.Textbox(label="P5 — gated", lines=6, interactive=False)
            p3_out = gr.Textbox(label="P3 — closed book", lines=6, interactive=False)
            gold_out = gr.Textbox(label="Correct answer", lines=6, interactive=False)
        agreement_out = gr.HTML()
        identical_out = gr.Markdown()
        gr.Markdown("### Evidence")
        chunks_out = gr.HTML()

        # ── callbacks ──────────────────────────────────────────────

        def _run(q, a, b, c, d, g, gate, th, tm, k):
            # A generator, so the first yield lands in the browser before the
            # model starts. One question is ~30 s of blocking CPU work with no
            # streaming inside it; without this the page sits silent throughout.
            yield ("⏳ Running — draft, gate, retrieve if it fires, then answer "
                   "twice. ~20 s skipping, ~60-95 s retrieving.",
                   "", "", "", "", "", "", "", "", "")

            choices = {letter: text.strip() for letter, text in
                       (("A", a), ("B", b), ("C", c), ("D", d))
                       if text and text.strip()}
            try:
                result = session.answer(
                    q, choices=choices or None, gate_type=gate,
                    entropy_threshold=th, margin_threshold=tm, top_k=int(k),
                    gold=g,
                )
            except Exception as exc:              # surfaced in the UI, not swallowed
                yield "", _error_html(str(exc)), "", "", "", "", "", "", "", ""
                return

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
                identical = ("*Retrieval was skipped, so both columns ran the same "
                             "closed-book prompt — identical by construction.*")

            gold_shown = ""
            if result.gold_letter:
                gold_shown = result.gold_letter
                if result.gold_text:
                    gold_shown += f". {result.gold_text}"

            chunks_html = (
                render_chunks_html(
                    result.chunks, result.query,
                    lexical_only=session.cfg.policy.retrieval_mode != "bm25",
                )
                if result.chunks
                else '<div class="mr-note">Retrieval skipped — no evidence '
                     "fetched.</div>"
            )
            yield (f"Done in {result.latency_s:.1f}s.",
                   _verdict_html(result),
                   render_gate_table_html(result.gate_details, result.gate_name),
                   heat, result.p5_answer, result.p3_answer, gold_shown,
                   render_agreement_html(result.p5_letter, result.p3_letter,
                                         result.gold_letter, result.p5_correct,
                                         result.p3_correct),
                   identical, chunks_html)

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
            [question, choice_a, choice_b, choice_c, choice_d, gold,
             gate_type, tau_h, tau_m, top_k],
            [status_out, verdict_out, gate_table_out, heatmap_out,
             p5_out, p3_out, gold_out, agreement_out, identical_out, chunks_out],
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
        # Base, not Soft/Default: the shipped themes bring an indigo accent and a
        # rounded display font. Everything visual here is set in CSS instead, so
        # the palette is one decision in one place.
        theme=gr.themes.Base(),
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        inbrowser=inbrowser,
    )
