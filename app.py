import os

import streamlit as st
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from pdf_processor import process_pdf

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Research Assistant RAG Chatbot", page_icon="📚")
st.title("📚 Research Assistant RAG Chatbot")
st.write("Upload your research paper and ask anything — including questions about **figures, graphs, and formulas**!")

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "documents" not in st.session_state:
    st.session_state.documents = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------------------------------------------------------
# LLM setup  (Llama 4 Scout – multimodal, free tier on Groq)
# ---------------------------------------------------------------------------
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.3,
)

# ---------------------------------------------------------------------------
# File upload & processing
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("📄 Upload a PDF", type=["pdf"])

if uploaded_file:
    # Only re-process if a new file is uploaded
    current_name = uploaded_file.name
    if st.session_state.get("uploaded_file_name") != current_name:
        pdf_path = "temp.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        # -- Progress UI --
        progress_bar = st.progress(0, text="Starting document processing...")
        status_text = st.empty()

        def update_progress(stage: str, current: int, total: int):
            """Callback fed to process_pdf for live progress updates."""
            pct = int((current / max(total, 1)) * 100)
            progress_bar.progress(pct, text=stage)
            status_text.caption(stage)

        # -- Run multimodal pipeline --
        with st.spinner("Processing your document..."):
            docs = process_pdf(pdf_path, llm, progress_callback=update_progress)

        progress_bar.progress(100, text="✅ Document processed!")
        status_text.empty()

        # -- Show extraction summary --
        text_count = sum(1 for d in docs if d.metadata.get("content_type") == "text")
        image_count = sum(
            1 for d in docs if d.metadata.get("content_type") == "image_caption"
        )
        st.success(
            f"Processed **{current_name}**: "
            f"{text_count} text chunks and {image_count} figure descriptions extracted."
        )

        # -- Build vector store --
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectordb = Chroma.from_documents(
            docs, embedding=embeddings, persist_directory="db"
        )
        st.session_state.retriever = vectordb.as_retriever(search_kwargs={"k": 5})
        st.session_state.documents = docs
        st.session_state.uploaded_file_name = current_name
        st.session_state.chat_history = []

# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------
if st.session_state.retriever:
    # Render chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # New user query
    query = st.chat_input("💬 Ask a question about your paper...")

    if query:
        # Display user message
        st.chat_message("user").markdown(query)
        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Retrieve relevant chunks (text + image captions)
                relevant_docs = st.session_state.retriever.invoke(query)
                context = "\n\n---\n\n".join(
                    [d.page_content for d in relevant_docs]
                )

                # Build augmented prompt
                prompt = (
                    "You are a helpful research assistant. Answer the question "
                    "based on the context below. The context includes text extracted "
                    "from a research paper, as well as descriptions of figures, "
                    "charts, graphs, and mathematical formulas found in the paper.\n\n"
                    "When answering:\n"
                    "• If the question is about a visual element (graph, figure, "
                    "chart), use the figure descriptions provided in the context.\n"
                    "• If the question involves a formula or equation, include the "
                    "LaTeX notation if available.\n"
                    "• Cite the page number when possible (e.g., 'On page 5...').\n"
                    "• If the context does not contain enough information, say so "
                    "honestly.\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question: {query}"
                )

                response = llm.invoke([HumanMessage(content=prompt)])

            st.markdown(response.content)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response.content}
            )

            # Show sources in an expander
            with st.expander("📖 Sources"):
                for i, doc in enumerate(relevant_docs, 1):
                    ctype = doc.metadata.get("content_type", "text")
                    page = doc.metadata.get("page", "?")
                    label = (
                        f"🖼️ Figure description (page {page})"
                        if ctype == "image_caption"
                        else f"📊 Table (page {page})"
                        if ctype == "table"
                        else f"📄 Text chunk (page {page})"
                    )
                    st.caption(f"**Source {i}:** {label}")
                    st.text(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))
                    st.divider()
