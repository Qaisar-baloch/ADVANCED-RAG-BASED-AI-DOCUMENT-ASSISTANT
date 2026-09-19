import io
import os
import re
import hashlib
import tempfile
from pathlib import Path

import faiss
import gdown
import numpy as np
import streamlit as st
from docx import Document
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -----------------------------
# App settings
# -----------------------------
st.set_page_config(page_title="AI Document Assistant", page_icon="📚", layout="wide")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 5
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"


# -----------------------------
# Document extraction
# -----------------------------
def extract_pdf(file_bytes, filename):
    """Extract one record per PDF page and handle encrypted PDFs safely."""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))

        # Some PDFs are encrypted even when they do not require a
        # password to open. Try the empty password first.
        if reader.is_encrypted:
            try:
                decrypt_result = reader.decrypt("")

                if decrypt_result == 0:
                    raise ValueError(
                        "This PDF is password-protected. "
                        "Please remove the password and upload it again."
                    )
            except ValueError:
                raise
            except Exception as exc:
                raise ValueError(
                    "This PDF is encrypted and could not be decrypted. "
                    "Make sure the PDF is not password-protected."
                ) from exc

        records = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            records.append(
                {
                    "filename": filename,
                    "page": page_number,
                    "text": text.strip(),
                }
            )

        return records

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"Could not read PDF '{filename}'. "
            "The file may be encrypted, corrupted, or use an unsupported PDF feature."
        ) from exc


def extract_docx(file_bytes, filename):
    """Extract text from a DOCX file."""
    document = Document(io.BytesIO(file_bytes))
    text = "\n".join(
        paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()
    )

    return [
        {
            "filename": filename,
            "page": None,
            "text": text.strip(),
        }
    ]


def extract_txt(file_bytes, filename):
    """Extract text from a TXT file."""
    text = file_bytes.decode("utf-8", errors="ignore")

    return [
        {
            "filename": filename,
            "page": None,
            "text": text.strip(),
        }
    ]


def extract_md(file_bytes, filename):
    """Extract text from a Markdown file."""
    text = file_bytes.decode("utf-8", errors="ignore")

    return [
        {
            "filename": filename,
            "page": None,
            "text": text.strip(),
        }
    ]


def extract_document(file_bytes, filename):
    """Choose the correct extractor from the file extension."""
    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file_bytes, filename)
    if extension == ".docx":
        return extract_docx(file_bytes, filename)
    if extension == ".txt":
        return extract_txt(file_bytes, filename)
    if extension == ".md":
        return extract_md(file_bytes, filename)

    return []


# -----------------------------
# Chunking
# -----------------------------
def chunk_documents(records, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split each extracted record into overlapping text chunks."""
    chunks = []
    step = max(1, chunk_size - overlap)

    for record in records:
        text = record["text"].strip()

        if not text:
            continue

        for start in range(0, len(text), step):
            chunk_text = text[start : start + chunk_size].strip()

            if chunk_text:
                chunks.append(
                    {
                        "filename": record["filename"],
                        "page": record["page"],
                        "text": chunk_text,
                    }
                )

            if start + chunk_size >= len(text):
                break

    return chunks


# -----------------------------
# Embeddings + FAISS
# -----------------------------
@st.cache_resource
def load_embedding_model():
    """Load the embedding model once per Streamlit process."""
    return SentenceTransformer(EMBEDDING_MODEL)


def build_vector_index(chunks):
    """Create embeddings and a FAISS cosine-similarity index."""
    model = load_embedding_model()

    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    # Inner product on normalized vectors = cosine similarity.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings


# -----------------------------
# Keyword search
# -----------------------------
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "i", "in", "is", "it", "of", "on", "or", "that", "the",
    "this", "to", "was", "what", "when", "where", "which", "who",
    "why", "with", "you", "your", "about", "does", "do", "can",
}


def important_words(text):
    """Return simple keywords from a question."""
    words = re.findall(r"\b[a-zA-Z0-9]{3,}\b", text.lower())
    return [word for word in words if word not in STOP_WORDS]


def keyword_score(question, chunk_text):
    """Score a chunk by the fraction of question keywords it contains."""
    keywords = set(important_words(question))

    if not keywords:
        return 0.0

    chunk_words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", chunk_text.lower()))
    matches = keywords.intersection(chunk_words)

    return len(matches) / len(keywords)


# -----------------------------
# Hybrid search
# -----------------------------
def hybrid_search(question, chunks, index, top_k=TOP_K):
    """Combine FAISS semantic search with keyword search."""
    model = load_embedding_model()

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    question_embedding = np.asarray(question_embedding, dtype="float32")

    # Get a wider semantic candidate set so keyword search can contribute.
    candidate_k = min(max(top_k * 3, 10), len(chunks))
    semantic_scores, semantic_ids = index.search(question_embedding, candidate_k)

    candidates = []

    for score, chunk_id in zip(semantic_scores[0], semantic_ids[0]):
        if chunk_id < 0:
            continue

        chunk = chunks[int(chunk_id)]
        candidates.append(
            {
                "chunk": chunk,
                "semantic_score": float(score),
                "keyword_score": keyword_score(question, chunk["text"]),
            }
        )

    # Hybrid score: semantic meaning is weighted more heavily.
    for item in candidates:
        item["hybrid_score"] = (
            0.70 * item["semantic_score"]
            + 0.30 * item["keyword_score"]
        )

    candidates.sort(key=lambda item: item["hybrid_score"], reverse=True)

    return candidates[:top_k]


# -----------------------------
# Google Drive loading
# -----------------------------
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


@st.cache_data(show_spinner=False)
def load_drive_files(url):
    """
    Download a public/shared Google Drive file or folder.

    gdown handles common Drive file and folder sharing URLs.
    The Drive item must be accessible without an interactive login.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Try folder download first.
        try:
            downloaded = gdown.download_folder(
                url,
                output=str(temp_path),
                quiet=True,
                use_cookies=False,
            )

            if downloaded:
                files = {}
                for path in temp_path.rglob("*"):
                    if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                        files[path.name] = path.read_bytes()

                if files:
                    return files
        except Exception:
            pass

        # If it was not a folder, try a single file.
        output_file = temp_path / "drive_file"
        downloaded_file = gdown.download(
            url,
            output=str(output_file),
            quiet=True,
            fuzzy=True,
            use_cookies=False,
        )

        if not downloaded_file:
            raise ValueError(
                "Could not download the Google Drive item. "
                "Make sure the link is shared and accessible."
            )

        # gdown may preserve the real filename in the returned path.
        downloaded_path = Path(downloaded_file)
        extension = downloaded_path.suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                "The Drive file type is not supported. "
                "Use PDF, DOCX, TXT, or MD."
            )

        return {downloaded_path.name: downloaded_path.read_bytes()}


# -----------------------------
# Utility functions
# -----------------------------
def file_fingerprint(files):
    """Create a stable key from filenames and file bytes."""
    hasher = hashlib.sha256()

    for filename, file_bytes in sorted(files.items()):
        hasher.update(filename.encode("utf-8"))
        hasher.update(file_bytes)

    return hasher.hexdigest()


def prepare_documents(files):
    """Extract and chunk all supported files.

    Returns:
        extracted: Successfully extracted document records.
        chunks: Text chunks created from successfully extracted records.
        errors: Per-file extraction errors.
    """
    extracted = []
    errors = []

    for filename, file_bytes in files.items():
        try:
            extracted.extend(extract_document(file_bytes, filename))
        except Exception as exc:
            errors.append(f"{filename}: {exc}")

    chunks = chunk_documents(extracted)

    return extracted, chunks, errors


def get_groq_client():
    """Read the Groq API key from Streamlit secrets."""
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        return None

    if not api_key:
        return None

    return Groq(api_key=api_key)


def answer_question(question, retrieved_chunks):
    """Ask Groq to answer only from retrieved document context."""
    client = get_groq_client()

    if client is None:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to "
            ".streamlit/secrets.toml or your Streamlit deployment secrets."
        )

    context_parts = []

    for number, item in enumerate(retrieved_chunks, start=1):
        chunk = item["chunk"]
        page_text = (
            f"page {chunk['page']}" if chunk["page"] is not None else "page not available"
        )

        context_parts.append(
            f"[Source {number}: {chunk['filename']}, {page_text}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """
You are an AI document assistant.

Answer the user's question ONLY using the provided document context.
Do not use outside knowledge.
If the answer is not present in the context, say:
"I could not find that information in the provided documents."

Be concise and clear.
"""

    user_prompt = f"""
DOCUMENT CONTEXT:
{context}

QUESTION:
{question}
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    return response.choices[0].message.content


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📚 AI Document Assistant")
st.write(
    "Upload documents or load them from Google Drive, then ask questions "
    "using semantic + keyword hybrid search."
)

# Session state keeps processed documents and embeddings between questions.
if "document_key" not in st.session_state:
    st.session_state.document_key = None
    st.session_state.extracted = []
    st.session_state.chunks = []
    st.session_state.index = None
    st.session_state.embeddings = None

with st.sidebar:
    st.header("Document sources")

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT or MD files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    st.divider()

    drive_url = st.text_input(
        "Google Drive file/folder link",
        placeholder="Paste a shared Google Drive link",
    )

    load_drive = st.button("Load Google Drive files")

    if load_drive and drive_url:
        try:
            with st.spinner("Loading Google Drive files..."):
                drive_files = load_drive_files(drive_url)

            # Save Drive files in session state so they remain available
            # when the user asks multiple questions.
            st.session_state.drive_files = drive_files
            st.success(f"Loaded {len(drive_files)} supported file(s).")
        except Exception as exc:
            st.error(str(exc))

    if "drive_files" not in st.session_state:
        st.session_state.drive_files = {}

# Collect local uploads.
local_files = {}
if uploaded_files:
    for uploaded_file in uploaded_files:
        local_files[uploaded_file.name] = uploaded_file.getvalue()

# Combine local + Drive documents.
all_files = {**st.session_state.drive_files, **local_files}

if all_files:
    current_key = file_fingerprint(all_files)

    # Only extract, chunk and embed when the document set changes.
    if current_key != st.session_state.document_key:
        with st.spinner("Extracting, chunking and embedding documents..."):
            extracted, chunks, extraction_errors = prepare_documents(all_files)

            if extraction_errors:
                for error in extraction_errors:
                    st.warning(f"Document skipped: {error}")

            if not chunks:
                st.error("No readable text was found in the selected documents.")
            else:
                index, embeddings = build_vector_index(chunks)

                st.session_state.document_key = current_key
                st.session_state.extracted = extracted
                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.embeddings = embeddings

                st.success("Documents processed and embeddings created.")

    if st.session_state.extracted:
        st.subheader("Extracted document information")

        document_info = []
        for filename, file_bytes in all_files.items():
            records = [
                record
                for record in st.session_state.extracted
                if record["filename"] == filename
            ]
            page_count = sum(
                1 for record in records if record["page"] is not None
            )

            document_info.append(
                {
                    "Filename": filename,
                    "Type": Path(filename).suffix.lower(),
                    "Pages": page_count if page_count else "N/A",
                    "Characters": sum(len(record["text"]) for record in records),
                }
            )

        st.dataframe(document_info, use_container_width=True, hide_index=True)
        st.info(f"Created {len(st.session_state.chunks)} overlapping text chunks.")

        with st.expander("Preview extracted text"):
            for record in st.session_state.extracted[:10]:
                page = (
                    f"Page {record['page']}"
                    if record["page"] is not None
                    else "Page not available"
                )
                st.markdown(f"**{record['filename']} — {page}**")
                st.write(record["text"][:1500] or "(No text extracted)")
                st.divider()

        st.subheader("Ask a question")

        question = st.text_input(
            "Question",
            placeholder="What does the document say about...?",
        )

        if question:
            if st.session_state.index is None:
                st.warning("Please load at least one readable document first.")
            else:
                with st.spinner("Searching documents..."):
                    results = hybrid_search(
                        question,
                        st.session_state.chunks,
                        st.session_state.index,
                        TOP_K,
                    )

                try:
                    with st.spinner("Generating answer..."):
                        answer = answer_question(question, results)

                    st.markdown("### Answer")
                    st.write(answer)

                    st.markdown("### Retrieved sources")

                    for number, item in enumerate(results, start=1):
                        chunk = item["chunk"]
                        page = (
                            str(chunk["page"])
                            if chunk["page"] is not None
                            else "N/A"
                        )

                        with st.expander(
                            f"{number}. {chunk['filename']} — Page {page}"
                        ):
                            st.caption(
                                f"Hybrid score: {item['hybrid_score']:.3f} | "
                                f"Semantic: {item['semantic_score']:.3f} | "
                                f"Keyword: {item['keyword_score']:.3f}"
                            )
                            st.write(chunk["text"])

                except Exception as exc:
                    st.error(str(exc))
else:
    st.info("Upload a document or load a Google Drive file/folder to begin.")
