# PA Policy Extraction Pipeline

Extracts Prior Authorization (PA) parameters for Plaque Psoriasis (PsO) from
payer policy PDFs and writes a structured `result.csv`.

---

## Quick Start (evaluator steps)

### 1 — Set credentials

```bash
cp .env.example .env
# Open .env and fill in GROQ_API_KEY (and optionally GROQ_MODEL).
```

The only required change is `GROQ_API_KEY`. Everything else has sensible defaults.

### 2 — Add inputs

| What | Where |
|---|---|
| Policy PDFs | `input_pdfs/` folder (create if absent) |
| Brand/filename list | `input.csv` in the project root |

`input.csv` must have at minimum two columns:

```
Filename,Brand
some_policy.pdf,SKYRIZI
other_policy.pdf,TREMFYA
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Run

```bash
python run_pipeline.py
```

Optional flag: `--rebuild-index` forces the PDF index to be rebuilt even if it
appears up-to-date (useful if you replaced PDFs without changing their names).

### 5 — Read results

```
output/result.csv
```

---

## Project layout

```
pa_pipeline/
├── run_pipeline.py        ← driver (run this)
├── chunking.py            ← PDF extraction + semantic chunking
├── indexing.py            ← FAISS index build / load
├── retrieval.py           ← Hybrid BM25 + dense retrieval, RRF, MMR
├── extraction.py          ← Prompt builder, LLM call, access score
├── requirements.txt
├── .env.example           ← copy to .env and fill in credentials
├── input.csv              ← (you provide) Filename + Brand rows
├── input_pdfs/            ← (you provide) source PDFs
├── output/
│   └── result.csv         ← final output
└── faiss_store/           ← auto-created index artefacts
    ├── policy_index.faiss
    ├── policy_metadata.pkl
    └── all_policy_chunks.csv
```

---

## Output columns

| Column | Description |
|---|---|
| Filename / Brand | Input identifiers |
| Age | Minimum age criterion (e.g. `>=6`, `NA`) |
| Step Therapy Requirements Documented in Policy | Verbatim step-therapy text |
| Number of Steps through Brands | Integer count or `NA` |
| Number of Steps through Generic | Integer count or `NA` |
| Step through-Phototherapy | `Yes` / `No` / `NA` |
| TB Test required | `Y` / `N` |
| Quantity Limits | Verbatim QL block or `NA` |
| Specialist Types | Comma-separated list or `NA` |
| Initial Authorization Duration(in-months) | Integer or `Unspecified` |
| Reauthorization Duration(in-months) | Integer, `Unspecified`, or `NA` |
| Reauthorization Required | `Yes` / `No` |
| Reauthorization Requirements Documented in Policy | Verbatim criteria or `NA` |
| Access Score | 0–100 (100 = most accessible) |
| Chunk IDs Used | Traceability: which chunks fed the LLM |
| Chunks Used (content) | Full chunk text for audit |
| Raw Response | Raw LLM output |
| Status | `success`, `json_parse_failed`, or `error: …` |

---

## .env reference

```dotenv
GROQ_API_KEY=          # ← required
GROQ_MODEL=llama-3.3-70b-versatile

INPUT_PDF_FOLDER=input_pdfs
INPUT_CSV=input.csv
OUTPUT_CSV=output/result.csv
FAISS_STORE=faiss_store

MIN_CHUNK_SIZE=250
MAX_CHUNK_SIZE=800
OVERLAP_TOKENS=150
MERGE_SIMILARITY_THRESHOLD=0.80

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

DENSE_TOP_K=30
BM25_TOP_K=30
RRF_K=60
MMR_TOP_K=3
MMR_LAMBDA=0.7
GLOBAL_TOP_CHUNKS=5

SLEEP_BETWEEN_ROWS=2
```

---
Hosted on : https://huggingface.co/spaces/Hiyaj/medical_payer_policy_intelligence
## Notes

- The index is rebuilt automatically when PDFs are added or modified. Pass
  `--rebuild-index` to force a full rebuild.
- Results are checkpointed to `output/result.csv` after every row, so a partial
  run is never lost.
- All extraction is scoped to **Plaque Psoriasis (PsO)** — other indications
  (PsA, UC, RA, etc.) are ignored by the LLM prompt.
