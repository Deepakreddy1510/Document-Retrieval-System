# Document Retrieval System 

A simple multi-document Retrieval-Augmented Generation application for asking questions over PDF files.

The project extracts text from PDFs, splits it into page-aware chunks, creates embeddings, stores the chunks and vectors in PostgreSQL with pgvector, retrieves relevant passages for a question, reranks them with a CrossEncoder, and sends only the best evidence to an LLM for grounded answer generation.
---

## What the project does

- Upload and index multiple PDF documents
- Extract text page by page with PyMuPDF
- Create tokenizer-aware chunks without crossing page boundaries
- Generate local embeddings with `BAAI/bge-small-en-v1.5`
- Store document metadata, chunks, page numbers, and vectors in PostgreSQL
- Retrieve relevant chunks with exact pgvector cosine search
- Rerank retrieval candidates with a CrossEncoder
- Generate answers with Groq or Gemini
- Add source labels such as `[S1]`, `[S2]`
- Show document and page-level evidence in the UI
- Refuse questions when the uploaded documents do not provide enough evidence
- Prevent the same PDF content from being indexed twice
- Evaluate retrieval separately from answer generation

---

## Architecture

```text
                           INGESTION

PDF upload
    |
    v
PyMuPDF text extraction
    |
    v
Minimal text cleaning
    |
    v
Page-aware token chunking
    |
    v
BAAI/bge-small-en-v1.5
384-dimensional embeddings
    |
    v
PostgreSQL + pgvector
(documents, chunks, page metadata, vectors)


                            QUESTION

User question
    |
    v
BGE query embedding
    |
    v
Exact dense retrieval with pgvector
Top-N candidates
    |
    v
CrossEncoder reranking
cross-encoder/ms-marco-MiniLM-L-6-v2
    |
    v
Final Top-K passages
    |
    v
Source-labelled context
[S1], [S2], ...
    |
    v
Groq / Gemini
    |
    v
Grounded answer + page-level evidence
```

The live V1 query path uses **dense retrieval followed by CrossEncoder reranking**.

PostgreSQL full-text retrieval and Reciprocal Rank Fusion (RRF) are still implemented in the repository because they were tested during retrieval experiments, but they are not the default live retrieval path.

---

## Core features

### Multi-document PDF ingestion

Each PDF is processed independently while retaining:

- original filename
- physical PDF page number
- chunk index
- chunk text
- embedding vector

This makes it possible to retrieve from several unrelated PDFs while keeping citations traceable to the correct document and page.

### Content-based duplicate detection

A SHA-256 hash is calculated from the PDF bytes before ingestion.

If the same PDF content is uploaded again, the pipeline returns:

```text
ALREADY_INDEXED
```

instead of inserting another copy of the document and its chunks.

### Grounded answers

Only the final reranked passages are sent to the LLM.

The prompt instructs the model to use the supplied evidence and cite source labels such as:

```text
[S1]
[S2]
```

The UI maps those labels back to the PDF filename, page number, and retrieved passage.

### Unsupported-question refusal

When the evidence is insufficient, the assistant is instructed to return:

```text
I couldn't find enough evidence in the uploaded documents to answer that.
```

The UI does not display retrieved candidates as evidence when the assistant refuses the question.

---

## Tech stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| Language | Python 3.11 |
| PDF extraction | PyMuPDF |
| Embeddings | Sentence Transformers |
| Embedding model | `BAAI/bge-small-en-v1.5` |
| Embedding size | 384 dimensions |
| Database | PostgreSQL |
| Vector search | pgvector |
| Lexical experiment | PostgreSQL Full-Text Search |
| Fusion experiment | Reciprocal Rank Fusion |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generation | Groq or Gemini |
| Local database runtime | Docker Compose |
| Testing | pytest |

---

## Ingestion and chunking

PDF text is extracted page by page.

Chunks never cross a page boundary, which keeps page citations straightforward.

Default chunking settings:

```text
Chunk size: 450 embedding-model tokens
Overlap:    60 tokens
```

The chunker uses the tokenizer associated with the embedding model rather than estimating tokens from characters.

The embedding model has a 512-token sequence limit, so the pipeline checks the configured chunk size against the model limit before indexing.

---

## Embeddings

The project uses:

```text
BAAI/bge-small-en-v1.5
```

Each chunk becomes a 384-dimensional vector.

Document chunks are embedded once during ingestion and stored in PostgreSQL.

For a user question, the same embedding model creates a query vector. A configured BGE query prefix is applied only to query embeddings.

---

## Retrieval

### Production retrieval

The live application uses exact dense retrieval:

```text
Question
  -> query embedding
  -> pgvector cosine search
  -> Top-N candidates
  -> CrossEncoder reranking
  -> final Top-K passages
```

pgvector uses cosine-distance ordering with:

```sql
<=>
```

No approximate-nearest-neighbour index is used in V1. Exact search keeps the baseline easy to reason about and works well for the current corpus size.

### Lexical retrieval experiment

The repository also contains PostgreSQL full-text retrieval using:

- `websearch_to_tsquery`
- `TSVECTOR`
- GIN index
- `ts_rank_cd`

This is PostgreSQL full-text ranking, not BM25.

### RRF experiment

Dense scores and full-text-search scores are on different scales, so the experimental hybrid path combines rankings using Reciprocal Rank Fusion:

```text
RRF(d) = sum(1 / (k + rank_r(d)))
```

The default RRF constant is:

```text
k = 60
```

---

## Reranking

Dense retrieval is fast enough to retrieve a larger candidate set, but vector similarity alone is not always the best final ranking.

The project reranks candidates with:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

A CrossEncoder scores each `(question, passage)` pair together.

Current default flow:

```text
Dense Top-20
    |
    v
CrossEncoder
    |
    v
Final Top-5
```

The reranker scores are ranking scores, not calibrated probabilities.

---

## Why dense + reranking is the default

The retrieval evaluation compares five configurations on the current human-reviewed benchmark:

| Configuration | Hit@5 | Precision@5 | Recall@5 | MRR | Median latency |
|---|---:|---:|---:|---:|---:|
| Dense | 1.000 | 0.415 | 0.831 | 0.962 | 88.1 ms |
| Lexical | 0.231 | 0.077 | 0.069 | 0.231 | 50.8 ms |
| Hybrid | 0.923 | 0.400 | 0.792 | 0.923 | 145.2 ms |
| Dense + reranker | 1.000 | 0.400 | **0.842** | **1.000** | 1448.8 ms |
| Hybrid + reranker | 1.000 | 0.385 | 0.826 | **1.000** | 1597.7 ms |

For this corpus and benchmark, dense retrieval followed by reranking provided the strongest overall result while keeping the live retrieval path simpler than hybrid retrieval.

These numbers describe this benchmark only; they are not intended as a general claim that dense search is always better than hybrid search.

---

## Generation

The LLM is used only after retrieval and reranking.

The generation layer supports two providers:

- Groq
- Gemini

Changing the generation provider does not change the retrieval pipeline.

The current development configuration uses Groq with:

```text
qwen/qwen3.6-27b
```

The generator receives:

```text
grounding instructions
+
retrieved source-labelled passages
+
user question
```

and returns a grounded answer containing `[S#]` citations.

---

## Evaluation

Retrieval and generation are evaluated separately.

### Retrieval evaluation

```bash
python evaluation/retrieval_eval.py --dataset evaluation/dataset.jsonl --k 5
```

The script compares:

- dense
- lexical
- hybrid
- dense + reranker
- hybrid + reranker

Metrics:

- Hit@K
- Precision@K
- Recall@K
- MRR
- median latency

Detailed results are written to:

```text
evaluation/retrieval_results.json
```

### Generation evaluation

```bash
python evaluation/generation_eval.py --dataset evaluation/dataset.jsonl
```

V1 automatically checks:

- citation presence on answerable questions
- refusal accuracy on unanswerable questions

The current benchmark produced:

```text
Citation presence on answerable questions: 100%
Refusal accuracy on unanswerable questions: 100%
```

The script also exports generated answers and retrieved source information for manual review of:

- faithfulness to retrieved context
- whether the answer addresses the question
- whether cited evidence supports the answer

---

## Database design

### `documents`

Stores document-level metadata:

```text
id
filename
file_hash
page_count
created_at
```

### `chunks`

Stores retrieval units:

```text
id
document_id
page_number
chunk_index
text
embedding VECTOR(384)
generated TSVECTOR
```

`document_id` links each chunk to its parent PDF.

Because `page_number` is stored directly on every chunk, the application can create page-level citations without maintaining a separate citation database.

---

## Reliability and edge cases

### Duplicate PDFs

The pipeline calculates a SHA-256 hash before indexing. Re-uploading the same document returns `already_indexed` instead of duplicating its chunks.

### Empty extraction

If a PDF produces no usable chunks, ingestion fails instead of storing an empty document.

### Embedding token limit

The configured chunk size is checked against the embedding model's maximum sequence length to avoid silent truncation during embedding.

### Unsupported questions

The generation prompt requires refusal when retrieved evidence does not support an answer.

### Multi-document retrieval

Every result keeps its document ID, filename, page number, and chunk information, so evidence from unrelated PDFs remains traceable.

### API or database errors

Errors are surfaced by the Streamlit application rather than silently returning fabricated answers.

### Scanned PDFs

V1 expects PDFs with extractable text. OCR is not currently included.

---

## Project structure

```text
pdf-rag-assistant/
├── app.py
├── .streamlit/
│   └── config.toml
├── src/
│   └── pdf_rag/
│       ├── chunking.py
│       ├── config.py
│       ├── context.py
│       ├── database.py
│       ├── embeddings.py
│       ├── evaluation.py
│       ├── fusion.py
│       ├── generation.py
│       ├── models.py
│       ├── pdf_processor.py
│       ├── pipeline.py
│       ├── reranker.py
│       └── retrieval.py
├── evaluation/
│   ├── dataset.jsonl
│   ├── generation_eval.py
│   └── retrieval_eval.py
├── scripts/
│   ├── ingest.py
│   ├── init_db.py
│   └── search.py
├── tests/
├── sql/
│   └── schema.sql
├── sample_docs/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Prerequisites

Install:

- Python 3.11
- Docker Desktop
- Git

You also need an API key for at least one supported generation provider:

- Groq
- Gemini

---

## Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/pdf-rag-assistant.git
cd pdf-rag-assistant
```

---

## Local setup on Windows

The project can run Python locally while PostgreSQL runs in Docker.

### 1. Start PostgreSQL

```cmd
docker compose up -d db
```

### 2. Create the Python environment

```cmd
py -3.11 -m venv .venv
```

### 3. Activate it

Windows CMD:

```cmd
.venv\Scripts\activate.bat
```

### 4. Install dependencies

```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure environment variables

Copy:

```cmd
copy .env.example .env
```

Example Groq configuration:

```env
DATABASE_URL=postgresql://rag:rag@127.0.0.1:5433/ragdb

LLM_PROVIDER=groq
LLM_MODEL=qwen/qwen3.6-27b
GROQ_API_KEY=your_groq_api_key

GEMINI_API_KEY=
```

Do not commit `.env`.

### 6. Initialize the database

```cmd
python scripts/init_db.py
```

### 7. Run tests

```cmd
pytest -q
```

### 8. Start the application

```cmd
streamlit run app.py
```

Open in your browser:

```text
http://localhost:8501
```

---

## Ingest a PDF from the command line

```cmd
python scripts/ingest.py "sample_docs\your-document.pdf"
```

Example output:

```text
INDEXED your-document.pdf | pages=120 chunks=145
```

If it has already been indexed:

```text
ALREADY_INDEXED your-document.pdf
```

---

## Inspect retrieval without generation

```cmd
python scripts/search.py "your question"
```

This is useful when debugging retrieval independently of the LLM.

---

## Run the evaluations

Retrieval:

```cmd
python evaluation/retrieval_eval.py --dataset evaluation/dataset.jsonl --k 5
```

Generation:

```cmd
python evaluation/generation_eval.py --dataset evaluation/dataset.jsonl
```

---

## Docker

To start the services defined in Docker Compose:

```cmd
docker compose up --build
```

To stop them:

```cmd
docker compose down
```

To inspect the database directly:

```cmd
docker compose exec db psql -U rag -d ragdb
```

---

## Current limitations

- no OCR for image-only/scanned PDFs
- no persistent conversation database
- no user accounts or permissions
- no ANN/HNSW vector index
- no automatic query rewriting
- no calibrated runtime relevance threshold yet
- CrossEncoder reranking is CPU-expensive compared with dense retrieval alone

These are deliberate boundaries for the current version rather than hidden assumptions.

---

## Possible future work

- contextual query rewriting for follow-up questions
- confidence/abstention calibration using labelled answerable and unanswerable examples
- chunk-size and overlap experiments
- embedding-model comparison
- HNSW versus exact-search benchmarking on a larger corpus
- OCR fallback for scanned PDFs
- layout-aware/table-aware extraction
- stronger generation evaluation
- public deployment and reproducible benchmark reports
