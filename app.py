import io
import os
import hashlib
import sqlite3
from datetime import datetime

import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 38px;
    font-weight: 800;
}

.subtitle {
    color: #6b7280;
    font-size: 16px;
}

.source-box {
    background-color: #f1f5f9;
    padding: 12px;
    border-radius: 8px;
    border-left: 4px solid #2563eb;
    margin-bottom: 8px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">📚 AI PDF RAG Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a PDF, search its content, and ask questions.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


model = load_model()


# ============================================================
# CHROMADB
# ============================================================

@st.cache_resource
def load_collection():

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_or_create_collection(
        name="documents"
    )

    return collection


collection = load_collection()


# ============================================================
# SQLITE CHAT HISTORY
# ============================================================

db = sqlite3.connect(
    "chat_history.db",
    check_same_thread=False
)

cursor = db.cursor()


cursor.execute("""
CREATE TABLE IF NOT EXISTS chats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    created_at TEXT
)
""")


cursor.execute("""
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER,
    role TEXT,
    content TEXT,
    created_at TEXT
)
""")


cursor.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_hash TEXT UNIQUE,
    filename TEXT,
    pages INTEGER,
    chunks INTEGER,
    uploaded_at TEXT
)
""")


db.commit()


# ============================================================
# CHAT FUNCTIONS
# ============================================================

def create_chat():

    cursor.execute(
        """
        INSERT INTO chats
        (title, created_at)
        VALUES (?, ?)
        """,
        (
            "New Chat",
            datetime.now().isoformat()
        )
    )

    db.commit()

    return cursor.lastrowid


def get_chats():

    cursor.execute(
        """
        SELECT *
        FROM chats
        ORDER BY id DESC
        """
    )

    return cursor.fetchall()


def get_messages(chat_id):

    cursor.execute(
        """
        SELECT role, content
        FROM messages
        WHERE chat_id = ?
        ORDER BY id ASC
        """,
        (chat_id,)
    )

    return cursor.fetchall()


def save_message(
    chat_id,
    role,
    content
):

    cursor.execute(
        """
        INSERT INTO messages
        (chat_id, role, content, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            chat_id,
            role,
            content,
            datetime.now().isoformat()
        )
    )

    db.commit()


def update_chat_title(
    chat_id,
    title
):

    cursor.execute(
        """
        UPDATE chats
        SET title = ?
        WHERE id = ?
        """,
        (
            title,
            chat_id
        )
    )

    db.commit()


def delete_chat(chat_id):

    cursor.execute(
        """
        DELETE FROM messages
        WHERE chat_id = ?
        """,
        (chat_id,)
    )

    cursor.execute(
        """
        DELETE FROM chats
        WHERE id = ?
        """,
        (chat_id,)
    )

    db.commit()


# ============================================================
# DOCUMENT FUNCTIONS
# ============================================================

def file_hash(file_bytes):

    return hashlib.sha256(
        file_bytes
    ).hexdigest()


def document_already_exists(
    document_hash
):

    cursor.execute(
        """
        SELECT *
        FROM documents
        WHERE file_hash = ?
        """,
        (document_hash,)
    )

    return cursor.fetchone()


def save_document(
    document_hash,
    filename,
    pages,
    chunks
):

    cursor.execute(
        """
        INSERT OR IGNORE INTO documents
        (
            file_hash,
            filename,
            pages,
            chunks,
            uploaded_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            document_hash,
            filename,
            pages,
            chunks,
            datetime.now().isoformat()
        )
    )

    db.commit()


def get_documents():

    cursor.execute(
        """
        SELECT filename, pages, chunks
        FROM documents
        ORDER BY id DESC
        """
    )

    return cursor.fetchall()


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(
    text,
    chunk_size=800,
    overlap=100
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:

            chunks.append(chunk)

        start = end - overlap

    return chunks


# ============================================================
# PROCESS PDF
# ============================================================

def process_pdf(
    file_bytes,
    filename
):

    reader = PdfReader(
        io.BytesIO(file_bytes)
    )

    chunks = []
    metadata = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        page_text = page.extract_text()

        if not page_text:
            continue

        page_chunks = create_chunks(
            page_text
        )

        for chunk in page_chunks:

            chunks.append(chunk)

            metadata.append(
                {
                    "filename": filename,
                    "page": page_number
                }
            )

    return (
        chunks,
        metadata,
        len(reader.pages)
    )


# ============================================================
# STORE PDF
# ============================================================

def store_pdf(
    file_bytes,
    filename
):

    document_hash = file_hash(
        file_bytes
    )

    if document_already_exists(
        document_hash
    ):

        return False, 0, 0

    (
        chunks,
        metadata,
        pages
    ) = process_pdf(
        file_bytes,
        filename
    )

    if not chunks:

        return False, 0, pages

    embeddings = model.encode(
        chunks
    )

    ids = [
        f"{document_hash}_{i}"
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadata
    )

    save_document(
        document_hash,
        filename,
        pages,
        len(chunks)
    )

    return True, len(chunks), pages


# ============================================================
# SEARCH
# ============================================================

def search_documents(
    question,
    number_of_results=3
):

    question_embedding = model.encode(
        [question]
    )

    results = collection.query(
        query_embeddings=
        question_embedding.tolist(),

        n_results=number_of_results
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    return documents, metadatas


# ============================================================
# SESSION CHAT
# ============================================================

if "chat_id" not in st.session_state:

    chats = get_chats()

    if chats:

        st.session_state.chat_id = chats[0][0]

    else:

        st.session_state.chat_id = create_chat()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("💬 Chat History")

    if st.button(
        "➕ New Chat",
        use_container_width=True
    ):

        st.session_state.chat_id = create_chat()

        st.rerun()


    chats = get_chats()

    for chat in chats:

        chat_id = chat[0]
        title = chat[1]

        if st.button(
            f"💬 {title}",
            key=f"chat_{chat_id}",
            use_container_width=True
        ):

            st.session_state.chat_id = chat_id

            st.rerun()


    st.divider()

    st.header("📄 Documents")

    documents = get_documents()

    if documents:

        for document in documents:

            filename = document[0]
            pages = document[1]
            chunks = document[2]

            st.write(
                f"📄 **{filename}**"
            )

            st.caption(
                f"{pages} pages • {chunks} chunks"
            )

    else:

        st.info(
            "No PDF uploaded yet."
        )


# ============================================================
# PDF UPLOAD
# ============================================================

st.subheader("📤 Upload PDF")

uploaded_files = st.file_uploader(
    "Choose one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


if uploaded_files:

    if st.button(
        "🚀 Process PDF"
    ):

        for uploaded_file in uploaded_files:

            with st.spinner(
                f"Processing {uploaded_file.name}..."
            ):

                file_bytes = (
                    uploaded_file.getvalue()
                )

                (
                    new_document,
                    chunks_count,
                    pages_count
                ) = store_pdf(
                    file_bytes,
                    uploaded_file.name
                )

                if new_document:

                    st.success(
                        f"✅ {uploaded_file.name} "
                        f"processed successfully!"
                    )

                    st.info(
                        f"📄 Pages: {pages_count} | "
                        f"🧩 Chunks: {chunks_count}"
                    )

                else:

                    st.info(
                        f"ℹ️ {uploaded_file.name} "
                        f"is already processed."
                    )


# ============================================================
# CHAT AREA
# ============================================================

st.divider()

st.subheader("🤖 Ask Questions")


messages = get_messages(
    st.session_state.chat_id
)


for role, content in messages:

    with st.chat_message(role):

        st.markdown(content)


# ============================================================
# USER QUESTION
# ============================================================

question = st.chat_input(
    "Ask something about your PDF..."
)


if question:

    save_message(
        st.session_state.chat_id,
        "user",
        question
    )

    with st.chat_message("user"):

        st.markdown(question)


    with st.chat_message("assistant"):

        if collection.count() == 0:

            answer = (
                "⚠️ Please upload and process "
                "a PDF before asking questions."
            )

            st.warning(answer)

        else:

            with st.spinner(
                "🔎 Searching your document..."
            ):

                documents, metadata = (
                    search_documents(
                        question,
                        3
                    )
                )

            if not documents:

                answer = (
                    "I couldn't find relevant "
                    "information in the uploaded PDF."
                )

                st.warning(answer)

            else:

                st.markdown(
                    "### 📖 Relevant Information"
                )

                for i, document in enumerate(
                    documents
                ):

                    st.markdown(
                        f"**Result {i + 1}**"
                    )

                    st.write(document)

                    if i < len(metadata):

                        st.markdown(
                            f"""
                            <div class="source-box">
                            📄 {metadata[i]["filename"]}
                            <br>
                            📖 Page {metadata[i]["page"]}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


    save_message(
        st.session_state.chat_id,
        "assistant",
        answer
    )


    current_messages = get_messages(
        st.session_state.chat_id
    )


    if len(current_messages) == 2:

        title = question[:40]

        update_chat_title(
            st.session_state.chat_id,
            title
        )


    st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "📚 AI PDF RAG Chatbot • "
    "Python + Streamlit + ChromaDB + "
    "Sentence Transformers"
)
