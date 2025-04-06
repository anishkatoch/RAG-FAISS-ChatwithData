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

## 🧪 Features

- ✅ Multi-format file support
- ✅ Responsive and clean Streamlit UI
- ✅ Memory summarization for context-aware conversations
- ✅ Embedding via OpenAI API
- ✅ FAISS vector search with top-k document retrieval

---

## 🖥️ UI Preview

![Smart Data Assistant UI Preview](https://user-images.githubusercontent.com/placeholder-image.png) <!-- Replace with actual screenshot URL -->

---

## 🔧 Setup Instructions

```bash
# Clone the repository
git clone https://github.com/your-username/smart-data-assistant.git
cd smart-data-assistant

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
touch .env
echo "OPENAI_API_KEY=your-key-here" >> .env

# Run the Streamlit app
streamlit run dataAssistant.py
