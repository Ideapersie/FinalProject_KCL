@echo off
REM ===================================================================
REM  run_overnight_a8.bat -- unattended: A8 gate variants + chunk-relevance re-run
REM
REM  Launch from a REAL terminal (double-click, or `!scripts\run_overnight_a8.bat`).
REM  It cannot be spawned from the agent's non-interactive shell.
REM
REM  Two jobs, in order:
REM    1. A8  -- measure the gate-ensemble variants (probe-weighted fix).
REM              ~20-40 min. Re-answers only the queries whose decision flips.
REM    2. P1 re-run WITH chunk logging (~1.5 h), then the chunk-relevance
REM              analysis that decides: is retrieval IRRELEVANT (retriever fault)
REM              or RELEVANT-BUT-DISTRACTING (model fault)? Those have opposite
REM              remedies, so the answer determines where effort goes next.
REM
REM  Both steps are RESUMABLE. Step 2's run_experiment skips question-ids already
REM  in its output file, so a kill costs minutes, not the night. The retry loops
REM  below relaunch until the work is actually done.
REM
REM  Everything is appended to results\raw_logs\overnight_a8.log.
REM ===================================================================

setlocal
cd /d "%~dp0.."

set LOG=results\raw_logs\overnight_a8.log
set MODEL=models\Llama-3.2-3B-Instruct-Q4_K_M.gguf
set BM25=indexes\bm25_medcorp_tp.pkl
set FAISS=indexes\faiss_medcorp_tp
set DATASET=data\raw\mirage\benchmark.json

echo. >> %LOG%
echo ================================================== >> %LOG%
echo  OVERNIGHT A8 + CHUNK RELEVANCE  %DATE% %TIME% >> %LOG%
echo ================================================== >> %LOG%

REM ---------------------------------------------------------------
REM  STEP 1 -- A8: gate-ensemble variants
REM  Not resumable in itself (it is short), so just retry on failure.
REM ---------------------------------------------------------------
REM  NOTE: A8 already completed (it took ~11 min, not hours) -- its results are in
REM  results\raw_logs\gate_variants_mcq.json. This step is now a no-op unless that
REM  file is deleted. The night's real work is step 2.
echo [1/3] A8 gate variants... >> %LOG%
echo [1/3] A8 gate variants...

if exist results\raw_logs\gate_variants_mcq.json (
  echo   already done -- skipping. >> %LOG%
  goto step2
)

:a8
python scripts\run_gate_variants.py ^
  --logs results\raw_logs\p5_medcorp_mcq.jsonl ^
  --dataset %DATASET% ^
  --bm25-index %BM25% --faiss-index %FAISS% ^
  --model %MODEL% ^
  --verify ^
  --output results\raw_logs\gate_variants_mcq.json >> %LOG% 2>&1
if exist results\raw_logs\gate_variants_mcq.json goto step2
echo   [retry] A8 did not finish; relaunching... >> %LOG%
goto a8

:step2
echo [1/3] A8 done. >> %LOG%

REM ---------------------------------------------------------------
REM  STEP 2 -- re-run P1 WITH retrieved chunks logged.
REM  The original P1 log predates chunk logging, so the chunks it was
REM  given were never recorded -- which is exactly what we now need in
REM  order to tell irrelevance apart from distraction.
REM  run_experiment skips question-ids already present in --output, so
REM  this loop simply resumes until all 200 are done.
REM ---------------------------------------------------------------
echo [2/3] P1 re-run with chunk logging (200 questions, ~1.5h)... >> %LOG%
echo [2/3] P1 re-run with chunk logging...

:p1
python scripts\_count.py results\raw_logs\p1_medcorp_mcq_chunks.jsonl 200 && goto step3
python scripts\run_experiment.py ^
  --policy configs\policies\p1_always_retrieve.yaml ^
  --experiment configs\experiments\mirage_medcorp.yaml ^
  --dataset %DATASET% ^
  --bm25-index %BM25% --faiss-index %FAISS% ^
  --model %MODEL% ^
  --output results\raw_logs\p1_medcorp_mcq_chunks.jsonl >> %LOG% 2>&1
echo   [resume] P1 chunk run continuing... >> %LOG%
goto p1

:step3
echo [2/3] P1 chunk re-run done (200/200). >> %LOG%

REM ---------------------------------------------------------------
REM  STEP 3 -- the analysis this was all for.
REM ---------------------------------------------------------------
echo [3/3] chunk-relevance analysis... >> %LOG%
echo [3/3] chunk-relevance analysis...

python scripts\analyse_chunk_relevance.py ^
  --p1 results\raw_logs\p1_medcorp_mcq_chunks.jsonl ^
  --p3 results\raw_logs\p3_mirage200.jsonl ^
  --dataset %DATASET% >> %LOG% 2>&1

echo. >> %LOG%
echo ============================================= >> %LOG%
echo  ALL OVERNIGHT WORK COMPLETE  %DATE% %TIME% >> %LOG%
echo ============================================= >> %LOG%
echo.
echo DONE. Results in %LOG%
echo   - A8 variants        : results\raw_logs\gate_variants_mcq.json
echo   - P1 with chunks     : results\raw_logs\p1_medcorp_mcq_chunks.jsonl
echo   - relevance verdict  : end of %LOG%

endlocal
