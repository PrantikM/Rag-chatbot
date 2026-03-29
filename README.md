# 📚 Research Assistant RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot built with Streamlit that lets you upload any research paper (PDF) and ask questions about it — powered by HuggingFace embeddings, ChromaDB, and Groq's blazing-fast LLMs.

---

## 🚀 Features

- 📄 Upload any PDF (research papers, documents, reports)
- 🔍 Semantic search over document content using vector embeddings
- 🤖 Context-aware answers via Groq's Mixtral / LLaMA3 LLMs
- 🧠 Local vector store with ChromaDB for fast retrieval
- ⚡ Simple, clean Streamlit UI — no frontend setup required

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| UI | [Streamlit](https://streamlit.io/) |
| PDF Parsing | LangChain `PyPDFLoader` |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | [ChromaDB](https://www.trychroma.com/) |
| LLM | [Groq](https://groq.com/) — `mixtral-8x7b-32768` / `llama3-70b-8192` |
| Orchestration | [LangChain](https://www.langchain.com/) |

---

## 📁 Project Structure

```
Rag-chatbot/
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
└── db/                  # ChromaDB persistent vector store (auto-generated)
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/PrantikM/Rag-chatbot.git
cd Rag-chatbot
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install streamlit langchain langchain-community langchain-groq \
            chromadb sentence-transformers pypdf numpy
```

### 4. Set your Groq API key

Get a free API key from [https://console.groq.com](https://console.groq.com), then export it:

```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

On Windows:
```cmd
set GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 🧩 How It Works

1. **Upload** a PDF via the Streamlit interface.
2. The PDF is **loaded and split** into chunks using `PyPDFLoader`.
3. Each chunk is **embedded** using `sentence-transformers/all-MiniLM-L6-v2` and stored in **ChromaDB**.
4. When you ask a question, the top-3 most **relevant chunks are retrieved** via similarity search.
5. The retrieved context + your question are sent to the **Groq LLM**, which returns a grounded answer.

```
User Query
    │
    ▼
[HuggingFace Embeddings]
    │
    ▼
[ChromaDB Vector Search] ──► Top-K Relevant Chunks
    │
    ▼
[Groq LLM (Mixtral / LLaMA3)] ──► Final Answer
```

---

## 📝 Environment Variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Your Groq API key (required) |

---

## 🔧 Configuration

You can swap the LLM model in `app.py` by changing the `model_name` parameter:

```python
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-70b-8192",   # or "mixtral-8x7b-32768"
    temperature=0.3,
)
```

---

## 📌 Requirements

- Python 3.9+
- A valid [Groq API key](https://console.groq.com)
- Internet connection (for downloading HuggingFace model on first run)

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open a [GitHub Issue](https://github.com/PrantikM/Rag-chatbot/issues) or submit a pull request.

---

## 📄 License

This project is open-source. Feel free to use and modify it for your own projects.

---

> Built with ❤️ by [PrantikM](https://github.com/PrantikM)
