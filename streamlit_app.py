
import os
import time
import hashlib
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Multi-PDF RAG Chatbot",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Chat with Your PDFs")
st.write(
    "Upload multiple PDF documents and ask questions "
    "about their contents."
)


# =========================================================
# 2. API CONFIGURATION
# =========================================================

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error(
        "GOOGLE_API_KEY was not found. "
        "Please check your .env file."
    )
    st.stop()


# =========================================================
# 3. MODELS
# =========================================================

@st.cache_resource
def get_embeddings(key):
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=key,
    )


@st.cache_resource
def get_llm(key):
    # Keep the model ID that works in your environment.
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=key,
        temperature=0,
    )


embeddings = get_embeddings(api_key)
llm = get_llm(api_key)


# =========================================================
# 4. SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = {}

if "file_hashes" not in st.session_state:
    st.session_state.file_hashes = {}


# =========================================================
# 5. PDF UPLOADER
# =========================================================

uploaded_files = st.file_uploader(
    "Upload your PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    key="pdf_uploader",
)


# =========================================================
# 6. HELPER: CREATE CHUNKS FROM A PDF
# =========================================================

def extract_pdf_chunks(file_name, file_bytes):
    """Extract text and metadata from a PDF and split it into chunks."""

    reader = PdfReader(BytesIO(file_bytes))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )

    chunks = []
    metadata = []
    readable_pages = 0

    for page_number, page in enumerate(reader.pages, start=1):

        page_text = page.extract_text() or ""

        if not page_text.strip():
            continue

        readable_pages += 1

        page_chunks = splitter.split_text(page_text)

        for chunk in page_chunks:
            chunks.append(chunk)

            metadata.append(
                {
                    "source": file_name,
                    "page": page_number,
                }
            )

    return chunks, metadata, readable_pages


# =========================================================
# 7. HELPER: CREATE EMBEDDINGS IN BATCHES
# =========================================================

def create_faiss_in_batches(chunks, metadata, batch_size=10):
    """
    Create a FAISS index in batches.
    Retry temporary API rate-limit errors with backoff.
    """

    if not chunks:
        raise ValueError("No readable text chunks were found.")

    vector_store = None

    for start in range(0, len(chunks), batch_size):

        end = min(start + batch_size, len(chunks))

        batch_chunks = chunks[start:end]
        batch_metadata = metadata[start:end]

        for attempt in range(4):

            try:
                batch_store = FAISS.from_texts(
                    texts=batch_chunks,
                    embedding=embeddings,
                    metadatas=batch_metadata,
                )

                if vector_store is None:
                    vector_store = batch_store
                else:
                    vector_store.merge_from(batch_store)

                break

            except Exception as error:

                error_text = str(error)

                is_rate_limit = (
                    "429" in error_text
                    or "RESOURCE_EXHAUSTED" in error_text
                )

                if not is_rate_limit or attempt == 3:
                    raise

                # Exponential backoff for temporary rate limits.
                wait_seconds = 15 * (2 ** attempt)

                st.warning(
                    f"API rate limit reached. "
                    f"Retrying in {wait_seconds} seconds..."
                )

                time.sleep(wait_seconds)

    return vector_store


# =========================================================
# 8. PROCESS UPLOADED FILES
# =========================================================

if uploaded_files:

    current_files = {}

    # Calculate a fingerprint for each file.
    for uploaded_file in uploaded_files:

        file_bytes = uploaded_file.getvalue()

        file_hash = hashlib.sha256(file_bytes).hexdigest()

        current_files[uploaded_file.name] = {
            "hash": file_hash,
            "bytes": file_bytes,
        }

    current_names = set(current_files.keys())
    previous_names = set(st.session_state.file_hashes.keys())

    # Remove PDFs that are no longer selected.
    removed_names = previous_names - current_names

    if removed_names:

        for name in removed_names:
            st.session_state.file_hashes.pop(name, None)
            st.session_state.processed_files.pop(name, None)

        # Rebuild a clean index for the remaining PDFs.
        st.session_state.vector_store = None
        st.session_state.messages = []

    # Identify new PDFs and PDFs whose contents changed.
    files_to_process = []

    for name, file_data in current_files.items():

        previous_hash = st.session_state.file_hashes.get(name)

        if previous_hash != file_data["hash"]:
            files_to_process.append(name)

    # Embed only new or modified PDFs when possible.
    if files_to_process:

        try:
            with st.status(
                "Processing your PDF documents...",
                expanded=True,
            ) as status:

                new_chunks = []
                new_metadata = []
                processed_names = []

                for name in files_to_process:

                    st.write(f"Reading: {name}")

                    chunks, metadata, page_count = extract_pdf_chunks(
                        name,
                        current_files[name]["bytes"],
                    )

                    if not chunks:
                        st.warning(
                            f"No readable text found in {name}."
                        )
                        continue

                    new_chunks.extend(chunks)
                    new_metadata.extend(metadata)

                    st.write(
                        f"Extracted {page_count} readable page(s) "
                        f"from {name}."
                    )

                    processed_names.append(name)

                if new_chunks:

                    st.write("Generating embeddings in batches...")

                    new_index = create_faiss_in_batches(
                        new_chunks,
                        new_metadata,
                        batch_size=10,
                    )

                    # Merge the new index with the existing one.
                    existing_index = st.session_state.vector_store

                    if existing_index is None:
                        st.session_state.vector_store = new_index
                    else:
                        existing_index.merge_from(new_index)

                        st.session_state.vector_store = existing_index

                    # Update hashes only after successful processing.
                    for name in processed_names:
                        st.session_state.file_hashes[name] = (
                            current_files[name]["hash"]
                        )

                    # A changed knowledge base may need new context.
                    st.session_state.messages = []

                status.update(
                    label="PDF processing completed!",
                    state="complete",
                    expanded=False,
                )

        except Exception as error:

            error_text = str(error)

            if (
                "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
            ):
                st.error(
                    "Gemini embedding quota exceeded. "
                    "Wait for the rate limit to reset, then try again."
                )

                st.caption(
                    "Reduce the number of chunks or check your Google "
                    "AI API usage and quota."
                )

            else:
                st.error("Could not process your PDFs.")
                st.exception(error)

    # Display selected files.
    st.success(
        f"📚 {len(uploaded_files)} PDF file(s) selected."
    )

    with st.expander("View uploaded PDFs"):

        for uploaded_file in uploaded_files:
            st.write(f"📄 {uploaded_file.name}")

    if st.session_state.vector_store is not None:
        st.success("Your PDF knowledge base is ready!")

else:

    st.info("Upload one or more PDFs to begin.")

    st.session_state.vector_store = None
    st.session_state.file_hashes = {}
    st.session_state.processed_files = {}
    st.session_state.messages = []


# =========================================================
# 9. CHAT INTERFACE
# =========================================================

if st.session_state.vector_store is not None:

    st.divider()

    st.subheader("💬 Chat with your PDFs")

    # Show existing conversation.
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask a question about your uploaded PDFs..."
    )

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Searching PDFs and generating an answer..."):

                try:
                    relevant_docs = (
                        st.session_state.vector_store
                        .similarity_search(
                            question,
                            k=4,
                        )
                    )

                    context_parts = []

                    for doc in relevant_docs:

                        source = doc.metadata.get(
                            "source",
                            "Unknown PDF",
                        )

                        page = doc.metadata.get("page", "?")

                        context_parts.append(
                            f"PDF: {source}\n"
                            f"Page: {page}\n"
                            f"Content: {doc.page_content}"
                        )

                    context = "\n\n".join(context_parts)

                    prompt = f"""
You are a helpful assistant that answers questions using PDF documents.

Rules:
1. Answer using the supplied PDF context.
2. If the answer is not present in the context, say so clearly.
3. Do not invent facts.
4. Explain the answer clearly.
5. Use information from multiple PDFs when relevant.

PDF CONTEXT:
{context}

USER QUESTION:
{question}
"""

                    response = llm.invoke(prompt)

                    answer = response.content

                    if isinstance(answer, list):

                        answer = "\n".join(
                            item.get("text", str(item))
                            if isinstance(item, dict)
                            else str(item)
                            for item in answer
                        )

                    answer = str(answer)

                    st.markdown(answer)

                    # Display source filenames and page numbers.
                    with st.expander("📌 Sources used"):

                        seen_sources = set()

                        for doc in relevant_docs:

                            source = doc.metadata.get(
                                "source",
                                "Unknown PDF",
                            )

                            page = doc.metadata.get("page", "?")

                            source_key = (source, page)

                            if source_key not in seen_sources:

                                seen_sources.add(source_key)

                                st.markdown(
                                    f"**PDF:** {source}  \n"
                                    f"**Page:** {page}"
                                )

                                st.caption(
                                    doc.page_content[:300]
                                    + (
                                        "..."
                                        if len(doc.page_content) > 300
                                        else ""
                                    )
                                )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )

                except Exception as error:

                    error_text = str(error)

                    if (
                        "429" in error_text
                        or "RESOURCE_EXHAUSTED" in error_text
                    ):
                        st.error(
                            "Gemini API quota exceeded. "
                            "Wait before trying again."
                        )

                    elif (
                        "503" in error_text
                        or "UNAVAILABLE" in error_text
                    ):
                        st.error(
                            "The AI service is temporarily unavailable. "
                            "Please try again shortly."
                        )

                    else:
                        st.error(f"Something went wrong: {error}")




