import os
import numpy as np
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage  # ✅ moved here in latest versions

# --- Streamlit UI ---
st.title("📚 Research Assistant RAG Chatbot")
st.write("Upload your research paper and ask anything about it!")

# --- File Upload ---
uploaded_file = st.file_uploader("📄 Upload a PDF", type=["pdf"])

if uploaded_file:
    pdf_path = "temp.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    st.info("Processing your document...")

    # --- Load and split PDF ---
    loader = PyPDFLoader(pdf_path)
    docs = loader.load_and_split()

    # --- Embeddings ---
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # --- Vector store ---
    vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory="db")
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # --- Groq LLM ---
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="mixtral-8x7b-32768",  # or "llama3-70b-8192"
        temperature=0.3,
    )

    # --- User Query ---
    query = st.text_input("💬 Ask a question about your paper:")

    if query:
        with st.spinner("Thinking..."):
            # Retrieve relevant chunks
            relevant_docs = retriever.get_relevant_documents(query)
            context = "\n\n".join([d.page_content for d in relevant_docs])

            # Build augmented prompt
            prompt = f"Answer the question based only on the context below.\n\nContext:\n{context}\n\nQuestion: {query}"

            # Get LLM response
            response = llm.invoke([HumanMessage(content=prompt)])

        st.success("✅ Answer:")
        st.write(response.content)
