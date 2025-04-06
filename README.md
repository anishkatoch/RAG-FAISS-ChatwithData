# RAG-FAISS-ChatwithData

# 📚 Smart Data Assistant

<div align="center">
  <img src="https://img.shields.io/badge/Powered%20By-GPT--4-green?style=flat-square&logo=openai" />
  <img src="https://img.shields.io/badge/Built%20With-Streamlit-orange?style=flat-square&logo=streamlit" />
  <img src="https://img.shields.io/badge/Tech%20Stack-LangChain%20%7C%20FAISS%20%7C%20OpenAI-blueviolet?style=flat-square" />
</div>

> ✨ Retrieval-Augmented Chat with GPT-4, FAISS & LangChain

---

## 🚀 About the Project

**Smart Data Assistant** is a Streamlit-based interactive chatbot that can:
- 📄 Ingest and understand PDF, DOCX, TXT, and PPTX files
- 💬 Chat intelligently with context-aware answers using GPT-4
- 🧠 Leverage vector embeddings and FAISS for fast semantic retrieval
- 🔄 Summarize previous chat history using LangChain's memory buffer

---
## 🖥️ Try It Live

👉 **[Click here to try Smart Data Assistant on Streamlit! 🚀](https://ragpoweredchatwithdata.streamlit.app)**

> No install needed – just upload your files and start chatting with your data.

---

> 🚀 **Click the badge above to launch the Smart Data Assistant!**

---

## 📂 File Upload Zone

Users can upload:
- 📝 `.txt`
- 📄 `.pdf`
- 🧾 `.docx`
- 📊 `.pptx`

Each file is parsed, processed, split, embedded using OpenAI embeddings, and stored in a **FAISS vector store**.

---

## 🧠 Tech Stack

| Tech | Description |
|------|-------------|
| 🧠 OpenAI GPT-4 | Natural language processing |
| 📚 LangChain | Chain-based LLM orchestration |
| 📁 FAISS | Efficient vector search |
| 📊 Streamlit | Frontend app interface |
| 🔐 dotenv | Secure environment variable management |

---

## 🔍 How It Works: RAG + FAISS

**Smart Data Assistant** uses **Retrieval-Augmented Generation (RAG)** to combine the power of GPT-4 with knowledge extracted from your uploaded documents.

---

### 🧠 RAG Flow Overview

**User Query → Search Chunks (FAISS) → Send Top Matches to GPT-4 → Get Answer**

1. **📄 File Parsing**  
   Upload PDFs, DOCXs, TXTs, or PPTXs — they’re read and converted into raw text.

2. **🔪 Chunking**  
   The text is split into smaller, meaningful chunks to preserve context.

3. **🧬 Embedding with OpenAI**  
   Each chunk is converted to a vector using OpenAI’s embedding model.

4. **📡 FAISS Vector Store**  
   These vectors are stored in a FAISS index to enable fast similarity-based retrieval.

5. **🔍 Real-Time Retrieval (RAG)**  
   When a user submits a query:
   - It’s embedded into a vector
   - FAISS retrieves the most relevant chunks
   - Chunks + query go into GPT-4 → 🤖 contextual, document-based answer

6. **🧠 Conversational Memory**  
   LangChain summarizes past interactions to maintain context through the conversation.

---

💡 This system enables GPT-4 to provide **accurate, source-grounded responses** — even across large, multi-document uploads.

---

## 🧪 Features

- ✅ Multi-format file support
- ✅ Responsive and clean Streamlit UI
- ✅ Memory summarization for context-aware conversations
- ✅ Embedding via OpenAI API
- ✅ FAISS vector search with top-k document retrieval

---

