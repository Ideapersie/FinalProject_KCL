# Final Report Design — 2026-08-04 (approved)

Target: ~14k body words (15k ceiling, body-only rule). All numbers via numbers.tex macros; drift guard untouched. No commits by Claude.

## Chapter plan
1. **Introduction (~1,300 w)** — Motivation reordered: safety → deployment reality (expanded: ITU 2025 stats, ~14% rural / 39% urban internet in low-income countries, generic phrasing, single laptop no GPU) → cost. Add RQ5 (scale transfer 3B→7B→14B). Contributions extended with scaling findings.
2. **Background (~2,100 w)** — §2.4 trimmed to pure literature; gate failure-mode/design rationale moves to Methodology. New refs: ITU 2025, WHO digital health / LMIC offline CDS, Med-PaLM 2, open medical LLMs, quantisation-calibration.
3. **Methodology (~2,500 w)** — absorbs moved gate rationale.
4. **Implementation (~1,400 w)** — light trim.
5. **Evaluation (~2,900 w)** — deep 3B study; forward-pointer to scaling chapter.
6. **Model Scaling (new, ~2,200 w)** — replaces qwen_scaling.tex + model_comparison.tex. Full 3×3 matrix, retrieval rate 50/32/24%, budget-matched recalibration, plateau 62→74.5→75.5, low bar at 7B / med bar unreachable, open inversion, break/fix 46/31/55, gate saturation. Tables: tab_scaling_results, tab_gate_saturation, tab_budget_match; fig_scaling.
7. **Discussion (~1,600 w)** — corpus-vs-distraction, threshold non-transfer synthesis, limitations (entropy off-by-one, StatPearls-subset corpus).
8. **LSEP (~900 w)**, 9. **Conclusion (~700 w)** — polish.

Mechanics: resolve duplicate labels from merged chapters; retire old chapters from main.tex \input list; abstract updated to lead with 3-model finding; references.bib gets researched real citations.
