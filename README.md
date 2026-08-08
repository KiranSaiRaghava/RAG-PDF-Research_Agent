# 📚 RAG PDF Research Agent

An AI-powered PDF Research Assistant that allows users to upload PDF documents, ask questions about their content, and perform web searches for current information.

The application combines **Retrieval-Augmented Generation (RAG)** with **web search** to provide more useful and up-to-date answers.

## 🚀 Live Demo

🔗 **Try the application:**  
https://rag-pdf-researchagent-hjrvjzprjpztdtb6jsjkoc.streamlit.app/

---

## 📌 Features

- 📄 Upload PDF documents
- 🔍 Search and retrieve relevant information from PDFs
- 🤖 Ask natural-language questions about uploaded documents
- 🌐 Search the web for recent/current information
- 🔗 Combine PDF knowledge with current web research
- 📑 Display PDF page references
- 💬 Interactive Streamlit chat interface
- 🧠 Retrieval-Augmented Generation (RAG)
- ⚡ AI-powered responses using Google Gemini
- 🔐 API keys securely managed using environment variables / Streamlit Secrets

---

## 🏗️ Architecture

```text
                    ┌──────────────────────┐
                    │      User            │
                    │ Upload PDF / Ask Q   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     Streamlit UI     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      AI Agent        │
                    │    LangChain Agent   │
                    └───────┬───────┬──────┘
                            │       │
                 ┌──────────┘       └──────────┐
                 ▼                             ▼
        ┌─────────────────┐          ┌─────────────────┐
        │   PDF Retrieval │          │  Tavily Search  │
        │      Tool       │          │      Tool       │
        └────────┬────────┘          └────────┬────────┘
                 │                            │
                 ▼                            ▼
        ┌─────────────────┐          ┌─────────────────┐
        │ Chroma Vector   │          │ Current Web     │
        │     Store       │          │ Information     │
        └────────┬────────┘          └────────┬────────┘
                 │                            │
                 └────────────┬───────────────┘
                              ▼
                    ┌──────────────────────┐
                    │    Gemini LLM        │
                    │   Final Response     │
                    └──────────────────────┘
