# FreightTiger AI Case Study: Smart Shipping Assistant

## Overview
This repository contains a Smart Assistant pipeline that tracks freight shipping costs by route, identifies abnormal price hikes, and uses a Retrieval-Augmented Generation (RAG) approach to determine if the hike is justified by real-world context notes.

## Core Logic & Baselines
- **Data Grouping:** Shipments are grouped by `route`, `route_type`, and `week_of` (standardized to Monday).
- **Cost Calculation:** Computed exactly as `total freight cost / (quantity * distance)`.
- **vs. Own History:** Uses a trailing 8-week rolling average. A `shift(1)` operation ensures strict prevention of look-ahead bias.
- **vs. Similar Routes:** Uses algebraic subtraction to calculate the average of all other routes of the same `route_type` in the same week, explicitly excluding the route itself.
- **Flagging Threshold:** Routes are flagged for review if their cost spikes by >5% against both their historical baseline and peer average.

## AI Integration & Hallucination Guardrails
To prevent hallucinations, the pipeline:
1. **Pre-filters Notes:** Only passes notes matching the specific route or "All Routes" to the context window.
2. **Zero-Temperature LLM:** Uses Google Gemini (`gemini-1.5-flash`) with `temperature=0.0` to eliminate creative variance and enforce strict logical deduction.
3. **Structured Prompts:** Instructs the LLM to ignore notes stating "costs were not affected" and forces a strict `JUSTIFIED / NOTE_ID / REASON` output format.

## Reproducibility Check
- **Methodology:** The script was executed 3 separate times on the same input dataset.
- **Result:** Identical output across all 3 runs (0 differences in flags, numerical values, or matched note IDs). The `temperature=0.0` setting ensures deterministic AI verdicts.

## Cost & Token Evaluation
- **Model Used:** `gemini-1.5-flash` (Free tier capable)
- **Estimated Input Tokens per LLM call:** ~150 tokens
- **Estimated Output Tokens per LLM call:** ~30 tokens
- **Total LLM Calls:** Depends on threshold spikes (approx. 40-50 calls for this dataset)
- **Total Cost Estimate:** < $0.01 per full run of the provided dataset based on published rates.

## How to Run
1. Install requirements: `pip install -r requirements.txt`
2. Update `main.py` with your API key.
3. Run the pipeline: `python main.py`
4. The exact required output will be generated as `submission_output.csv`.
