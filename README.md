# 📄 Chat with PDF using Gemini (LangChain + Streamlit)

An intelligent assistant that lets you upload PDF documents and ask questions about their contents using Google Gemini (Generative AI). Built using LangChain, FAISS for vector search, and Streamlit for the interface.

---

## 🧠 Tech Stack

| Tool | Purpose |
|------|---------|
| **Python** | Core programming language |
| **Streamlit** | UI and app framework |
| **LangChain** | Framework for building LLM apps |
| **FAISS** | Vector similarity search |
| **HuggingFace Embeddings** | Convert text to embeddings |
| **Google Gemini API** | LLM-based question answering |
| **dotenv** | Manage environment variables |
| **PyPDFLoader** | Read and parse PDF documents |

---

## 🚀 Features

- Upload multiple PDF documents.
- Automatically split and embed text using HuggingFace.
- Store embeddings using FAISS vector DB.
- Ask natural language questions about the content.
- Uses Google Gemini to generate intelligent responses.

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/pdf-assistant.git
cd pdf-assistant
