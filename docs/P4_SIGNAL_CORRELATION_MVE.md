# SYNTERA — P4 SIGNAL CORRELATION MVE
**STATUS: BLOCKED / REVISE**

## 1. Objective
To execute a Minimum Viable Experiment (MVE) calculating point-biserial correlations between extractable Syntera signals (Retrieval Margin, Evidence Density, RRF Gap, IDF) and downstream LLM generation success.

## 2. Infrastructure Inspection
Signals can be reliably extracted from the current pipeline:
*   **Dense Margin:** 	op_1_score - mean(top_2_to_k_scores) via Qdrant output.
*   **Evidence Density:** Structural aggregation via section_path, chunk_index, and page metadata populated by the P2 PyMuPDF parser.
*   **RRF Gap:** Rank variance available post-RRF fusion in etrieval/rag.py.

## 3. Data & Environment Blocker
Per the strict adversarial directive: *"If the environment cannot obtain an adequate dataset, STOP and report the blocker rather than fabricating one."*

**Blocker 1: Structural PDF Dependency**
Standard open-source QA datasets (SQuAD, NQ, HotpotQA) provide raw string contexts, not PDF files. Syntera's novel "Evidence Density" and "Chunk Continuity" signals depend entirely on physical layout metadata (e.g., bounding boxes, pages, section hierarchies) extracted by PyMuPDF. We cannot test these signals on raw text datasets without losing the primary features we seek to validate. 

**Blocker 2: Compute Constraints**
The final engineering benchmark (P3) established that the local HuggingFace LLM fallback (used when Ollama is unavailable) requires ~16 seconds per query on the CPU. A statistically significant MVE (e.g., N=500 queries) would require continuous compute for over 2.5 hours, risking timeout and rendering ad-hoc script execution impractical without batched GPU acceleration.

## 4. Falsification Conditions Triggered
*   **The dataset is too small or biased:** We only possess 3 PDFs (19 chunks). Training or drawing statistical correlation from 20 queries on a 19-chunk corpus is mathematically invalid.
*   **The experiment contains leakage:** The existing 20-query evaluation set was used for P1/P2/P3 architectural validation. It is fully leaked and cannot be used to evaluate a new correlation metric cleanly.

## 5. Final Decision
**REVISE**
The experiment design must be revised. To properly test these signals, Syntera requires a custom, structurally-parsed PDF dataset (e.g., downloading 50 random open-access ArXiv PDFs, parsing them via PyMuPDF, and generating a 500-query synthetic test set) executed on a GPU-enabled instance. Attempting to force the experiment on the current 19-chunk CPU environment violates the anti-fabrication mandate.
