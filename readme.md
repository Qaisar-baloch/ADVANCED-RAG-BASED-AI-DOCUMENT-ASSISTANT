# 📚 AI Document Assistant

A simple Streamlit RAG-style document assistant using:

- **Streamlit** for the user interface
- **PDF / DOCX / TXT / MD** text extraction
- **Overlapping text chunking**
- **Sentence Transformers** for embeddings
- **FAISS** for semantic vector search
- **Keyword search** for exact-word matching
- **Hybrid search** combining semantic + keyword scores
- **Groq** for question answering
- **Google Drive** as an additional document source

The code is intentionally kept in one `app.py` file so it is easy to study and explain.

## Project files

```text
ai_document_assistant/
├── app.py
├── requirements.txt
└── readme.md
```

## 1. Install dependencies

Create a virtual environment if desired:

```bash
python -m venv venv
```

Activate it:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install the packages:

```bash
pip install -r requirements.txt
```

## 2. Add the Groq API key

The API key is **not hardcoded in `app.py`**.

Create this file:

```text
.streamlit/secrets.toml
```

Put:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

Do not commit `secrets.toml` to Git.

For Streamlit Community Cloud, add the same key through the app's Secrets settings.

## 3. Run the app

```bash
streamlit run app.py
```

## How the app works

### Step 1 — Upload or load documents

The app accepts:

- PDF
- DOCX
- TXT
- Markdown (`.md`)

It also accepts a Google Drive file or folder link.

For Google Drive, the files need to be accessible to the downloader. In the simple version of this project, use a shared/public Drive link that does not require an interactive Google login.

### Step 2 — Extract text

There is a separate extraction function for each file type:

```text
extract_pdf()
extract_docx()
extract_txt()
extract_md()
```

PDFs are extracted page-by-page, so PDF chunks keep their page number.

DOCX, TXT and MD files keep the filename but use `N/A` for page because these formats do not reliably provide page information through the simple extractors used here.

### Step 3 — Chunk the text

Large documents are split into overlapping chunks.

Current settings:

```python
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
```

Every chunk keeps:

```text
filename
page
text
```

The app displays the total number of chunks created.

### Step 4 — Create embeddings

Sentence Transformers converts every chunk into a numerical vector.

The model is:

```text
all-MiniLM-L6-v2
```

The embeddings are normalized and stored in Streamlit session state.

The embedding model itself is loaded with:

```python
@st.cache_resource
```

This prevents the model from being loaded again unnecessarily.

### Step 5 — Build the FAISS index

FAISS stores the chunk embeddings.

The app uses:

```text
IndexFlatIP
```

Because the embeddings are normalized, inner product works as cosine similarity.

### Step 6 — Ask a question

The user's question is converted into an embedding.

FAISS retrieves semantically similar chunks.

At the same time, the app performs a simple keyword search.

Keyword score is based on how many important words from the question appear in the chunk.

### Step 7 — Hybrid ranking

The final score is:

```text
Hybrid Score =
    70% semantic score
  + 30% keyword score
```

This gives semantic search the larger role while still helping exact terms match.

### Step 8 — Groq answer

The top retrieved chunks are sent to Groq as context.

The system instruction tells the model:

- answer only from the supplied document context
- do not use outside knowledge
- say that the information is unavailable when it is not in the retrieved context

The current example uses:

```text
llama-3.3-70b-versatile
```

You can change `GROQ_MODEL` in `app.py` if you want to use another supported Groq model.

### Step 9 — Show sources

After every answer, the app shows the retrieved chunks.

Each source includes:

- filename
- page number when available
- retrieved text
- hybrid score
- semantic score
- keyword score

## Important optimization

The app does **not** create embeddings every time you ask a question.

The document set receives a SHA-256 fingerprint.

When the same documents are used again:

```text
document set unchanged
        ↓
reuse existing chunks
        ↓
reuse existing embeddings
        ↓
reuse existing FAISS index
        ↓
only search for the new question
```

When the uploaded/Drive documents change, the app rebuilds the extraction, chunks, embeddings and FAISS index.

This is handled using Streamlit `session_state`.

## Google Drive behavior

Paste a shared Google Drive file or folder link in the sidebar.

The app attempts to:

1. download a Drive folder
2. find supported files inside it
3. download a single Drive file if it is not a folder
4. pass the downloaded files into the exact same extraction/chunking/embedding/search pipeline

Supported Drive file extensions:

```text
.pdf
.docx
.txt
.md
```

Google Docs/Sheets/Slides are intentionally not included in this simple version.

## Security

Never put the Groq key directly in `app.py`.

Use:

```text
.streamlit/secrets.toml
```

with:

```toml
GROQ_API_KEY = "your-key"
```

Also add `.streamlit/secrets.toml` to `.gitignore`.

## Simple architecture

```text
                 ┌─────────────────┐
                 │ Local Uploads    │
                 └────────┬────────┘
                          │
                 ┌────────▼────────┐
                 │ Google Drive     │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Text Extraction │
                 └────────┬────────┘
                          ▼
                 ┌─────────────────┐
                 │ Chunking        │
                 └────────┬────────┘
                          ▼
                 ┌─────────────────┐
                 │ Embeddings      │
                 │ SentenceTrans.  │
                 └────────┬────────┘
                          ▼
                 ┌─────────────────┐
                 │ FAISS Index     │
                 └────────┬────────┘
                          │
             User Question
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
       Semantic Search          Keyword Search
              │                       │
              └───────────┬───────────┘
                          ▼
                   Hybrid Ranking
                          │
                          ▼
                  Retrieved Chunks
                          │
                          ▼
                       Groq
                          │
                          ▼
                       Answer
                          │
                          ▼
                       Sources
```

## Notes

This is a deliberately simple educational implementation. It is suitable for learning the main RAG pipeline without introducing a large framework such as LangChain or LlamaIndex.

For larger production document collections, you would normally move the vector index and document metadata into persistent storage instead of keeping them only in Streamlit session state.
