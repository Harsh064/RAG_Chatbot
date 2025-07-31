# 🧠 RAG Chatbot

Chat with your PDFS using **natural language**! 

    - This AI-powered RAG chatbot by which we can chat with PDFS in natural language

    - A Retrieval-Augmented Generation (RAG) chatbot built using Streamlit, LangChain, Gemini (Google Generative AI), and Hugging Face Embeddings. Upload PDF documents, process their content into vector embeddings, and interact with the chatbot to get context-aware answers based on your files.

    - Loom Videos to show :

            1.Working Demo -  https://www.loom.com/share/bba5d7b291674445abf7587d6f842c60?sid=727cd8c2-e0c8-4f73-823a-33743e7d9ac8

            2. Code Approach Explanation : https://www.loom.com/share/1a428f45e3b8448a9e50e98342362357?sid=f6b81dc1-711c-42cd-ae3a-8c85e5e532dc

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
## 🧰 Libraries Used & Why

| Library | Purpose |
|--------|---------|
| `streamlit` | Web interface for interacting with the chatbot |
| `dotenv` | Loads environment variables (like API keys) from `.env` file |
| `PyPDF2` | Reads and extracts text from PDF files |
| `langchain_text_splitters.CharacterTextSplitter` | Splits text into smaller overlapping chunks for better processing |
| `langchain_huggingface.embeddings.HuggingFaceEndpointEmbeddings` | Connects to Hugging Face Inference API to get embeddings (text → vector) |
| `langchain_community.vectorstores.FAISS` | Stores and searches the text vectors using Facebook’s FAISS |
| `langchain_google_genai.ChatGoogleGenerativeAI` | Gemini (Google) LLM to generate answers from the text |
| `langchain_core.messages` | Structures the chat (e.g., `HumanMessage`, `AIMessage`) |
| `langchain_core.prompts.ChatPromptTemplate` | Template for how the question and context are passed to the model |
| `langchain_core.runnables` | Builds modular chain components |
| `langchain_core.runnables.history.RunnableWithMessageHistory` | Adds session memory to keep chat history |
| `langchain.chains.combine_documents.create_stuff_documents_chain` | Combines retrieved documents and uses them to answer questions |
| `langchain.chains.create_retrieval_chain` | Combines retriever and LLM to form the RAG chain |
| `langchain_community.chat_message_histories.ChatMessageHistory` | Stores the message history in memory |
| `htmlTemplates` | Contains CSS and HTML templates for bot and user messages in the UI |

---
