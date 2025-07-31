# 🧠 RAG Chatbot

Chat with your PDFS using **natural language**! 

    - This AI-powered agent understands user queries from natural language

    - A Retrieval-Augmented Generation (RAG) chatbot built using Streamlit, LangChain, Gemini (Google Generative AI), and Hugging Face Embeddings. Upload PDF documents, process their content into vector embeddings, and interact with the chatbot to get context-aware answers based on your files.

    - Loom Videos to show working model: 
            1. https://www.loom.com/share/a3fae6c6d67a4346b8a15d4b1b5a0633?sid=62697a53-819a-4e3f-8d93-b6d104d5bf33
            2. https://www.loom.com/share/ae4291976e5e42c7978908b7ef3be8d3?sid=e57197a6-d41c-419b-ace0-b6903830feaf

---

## 🚀 Features

- Upload and process multiple PDF files
- Chunk and embed PDF content using HuggingFace Inference API
- Store embeddings in FAISS vector store
- Use Google Gemini-2.0 Flash for answering user queries
- Handles chat history via LangChain Message History
- Built-in prompt templates for retrieval-aware rephrasing and answering
- Modern chatbot UI with Streamlit

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Harsh064/RAG_Chatbot.git
cd RAG_PROJECT
```

### 2. Install Dependencies

We recommend using a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory:

```bash
GOOGLE_API_KEY="your-generative-ai-key"
HF_TOKEN="YOUR-HUGGINGFACE-api-key"
```

### 4. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---
#  How It Works
    1. PDF Upload
    Users upload one or more PDF files via the sidebar.

    2. Text Extraction
    Uses PyPDF2 to extract text from uploaded documents.

    3. Chunking
    Text is split into manageable chunks (1000 characters with 200 overlap).Text is split into manageable chunks (1000 characters with 200 overlap).

    4. Vector Embedding
    Chunks are embedded via HuggingFaceEndpointEmbeddings and stored in FAISS.

    5) Conversational Chain

    - Rewrites user queries into standalone questions using Gemini.

    - Retrieves relevant chunks using vector similarity.

    - Responds using Gemini with retrieved context and chat history.

    6) Chat History
    Stored using LangChain’s ChatMessageHistory to maintain session memory.

---

## 🧱 Components

### 1. Streamlit Frontend (`app.py`)
- Provides a chat-based user interface.
- Displays both user queries and agent responses.
- Handles session management and chat history.

### 2. HTML and CSS (`htmlTemplates.py`)
- html and css for UI

---

## 🗂 Directory Structure

```
├── app.py                      # Streamlit frontend app
├── htmlTemplates.py            # HTML/CSS for chatbot UI
├── .env                        # API KEYS
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation (you are here)
```

---

## 🔐 Environment Variables

Stored in a `.env` file:

- GOOGLE_API_KEY=your_google_genai_api_key
- HF_TOKEN=your-huggingface-api-key

---

