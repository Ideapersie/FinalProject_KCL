@echo off
REM ===================================================================
REM run_overnight.bat - real-corpus (MedCorp) policy re-run queue.
REM
REM Runs in its OWN OS process, independent of any Claude Code session,
REM so it survives session teardown. Every run is resumable (skips
REM already-logged question_ids), so if the machine sleeps or reboots you
REM can just launch this again and each step continues where it stopped.
REM
REM LAUNCH (from a REAL terminal, so it detaches from any Claude session):
REM     scripts\run_overnight.bat
REM   or double-click the file in Explorer. Leave the black window open.
REM
REM Keep the laptop on AC power. Close other apps first: the MedCorp FAISS
REM index (~654 MB) + BM25 (~787 MB) + the 3B model make this RAM-tight.
REM Progress -> results\raw_logs\*.jsonl and results\raw_logs\overnight.log.
REM ===================================================================

cd /d "%~dp0\.."
set PYTHONUNBUFFERED=1
set LOG=results\raw_logs\overnight.log

set MODEL=models\Llama-3.2-3B-Instruct-Q4_K_M.gguf
set BM25=indexes\bm25_medcorp_tp.pkl
set FAISS=indexes\faiss_medcorp_tp

echo ============================================ >> "%LOG%"
echo MedCorp re-run started %DATE% %TIME%         >> "%LOG%"
echo ============================================ >> "%LOG%"

REM Each block loops until its output has 200 records, relaunching on any
REM early exit (the resume logic makes re-entry safe and cheap).

REM ---- 1) P1 always-retrieve, MCQ, hybrid, real corpus -----------------
:p1_mcq
echo [%TIME%] P1 MCQ (medcorp) >> "%LOG%"
python scripts\run_experiment.py --policy configs\policies\p1_always_retrieve.yaml ^
  --experiment configs\experiments\mirage_medcorp.yaml --hardware configs\hardware_medium.yaml ^
  --dataset data\raw\mirage\benchmark.json --model %MODEL% ^
  --bm25-index %BM25% --faiss-index %FAISS% --retrieval-mode hybrid ^
  --max-questions 200 --output results\raw_logs\p1_medcorp_mcq.jsonl >> "%LOG%" 2>&1
python scripts\_count.py results\raw_logs\p1_medcorp_mcq.jsonl 200 || goto p1_mcq

REM ---- 2) P4 hybrid, MCQ, real corpus ---------------------------------
:p4_mcq
echo [%TIME%] P4 MCQ (medcorp) >> "%LOG%"
python scripts\run_experiment.py --policy configs\policies\p4_hybrid.yaml ^
  --experiment configs\experiments\mirage_medcorp.yaml --hardware configs\hardware_medium.yaml ^
  --dataset data\raw\mirage\benchmark.json --model %MODEL% ^
  --bm25-index %BM25% --faiss-index %FAISS% --retrieval-mode hybrid ^
  --max-questions 200 --output results\raw_logs\p4_medcorp_mcq.jsonl >> "%LOG%" 2>&1
python scripts\_count.py results\raw_logs\p4_medcorp_mcq.jsonl 200 || goto p4_mcq

REM ---- 3) P5 gated (calibrated), MCQ, real corpus ---------------------
:p5_mcq
echo [%TIME%] P5 MCQ (medcorp, calibrated) >> "%LOG%"
python scripts\run_experiment.py --policy configs\policies\p5_gated_entropy.yaml ^
  --experiment configs\experiments\mirage_p5_calibrated.yaml --hardware configs\hardware_medium.yaml ^
  --dataset data\raw\mirage\benchmark.json --model %MODEL% ^
  --bm25-index %BM25% --faiss-index %FAISS% --retrieval-mode hybrid ^
  --max-questions 200 --output results\raw_logs\p5_medcorp_mcq.jsonl >> "%LOG%" 2>&1
python scripts\_count.py results\raw_logs\p5_medcorp_mcq.jsonl 200 || goto p5_mcq

REM ---- 4) P1 always-retrieve, open-ended, real corpus ----------------
:p1_open
echo [%TIME%] P1 open (medcorp) >> "%LOG%"
python scripts\run_experiment.py --policy configs\policies\p1_always_retrieve.yaml ^
  --experiment configs\experiments\pubmedqa_open.yaml --hardware configs\hardware_medium.yaml ^
  --dataset data\raw\openqa\pubmedqa_labeled.jsonl --model %MODEL% ^
  --bm25-index %BM25% --faiss-index %FAISS% --retrieval-mode hybrid ^
  --max-questions 200 --output results\raw_logs\p1_medcorp_open.jsonl >> "%LOG%" 2>&1
python scripts\_count.py results\raw_logs\p1_medcorp_open.jsonl 200 || goto p1_open

REM ---- 5) P5 gated (calibrated), open-ended, real corpus -------------
:p5_open
echo [%TIME%] P5 open (medcorp, calibrated) >> "%LOG%"
python scripts\run_experiment.py --policy configs\policies\p5_gated_entropy.yaml ^
  --experiment configs\experiments\pubmedqa_open.yaml --hardware configs\hardware_medium.yaml ^
  --dataset data\raw\openqa\pubmedqa_labeled.jsonl --model %MODEL% ^
  --bm25-index %BM25% --faiss-index %FAISS% --retrieval-mode hybrid ^
  --max-questions 200 --output results\raw_logs\p5_medcorp_open.jsonl >> "%LOG%" 2>&1
python scripts\_count.py results\raw_logs\p5_medcorp_open.jsonl 200 || goto p5_open

echo [%TIME%] ALL MEDCORP RUNS COMPLETE >> "%LOG%"
echo. >> "%LOG%"
echo Done. All 5 runs complete. Results in results\raw_logs\medcorp_*.jsonl
echo You can close this window.
