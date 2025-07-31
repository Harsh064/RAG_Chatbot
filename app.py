import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory 
import os
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
google_api_key=os.getenv("GOOGLE_API_KEY")

from htmlTemplates import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    """
    Extracts text from a list of PDF documents.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
    return text


def get_text_chunks(text):
    """
    Splits a given text into smaller chunks for processing.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    """
    Creates a vector store from text chunks using Hugging Face embeddings.
    Requires HUGGINGFACEHUB_API_TOKEN environment variable to be set.
    """
    hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_api_token:
        st.error("HUGGINGFACEHUB_API_TOKEN environment variable is not set. Please set it in your .env file or system environment.")
        return None

    try:
        
        embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            task="feature-extraction",
            huggingfacehub_api_token=hf_api_token
        )
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store with Hugging Face Embeddings: {e}")
        st.info("Please check your HUGGINGFACEHUB_API_TOKEN and ensure the embedding model is accessible.")
        return None


# In-memory session store for chat history
store = {}
def get_session_history(session_id: str) -> ChatMessageHistory:
    """
    Returns a ChatMessageHistory instance for a given session ID.
    If the session doesn't exist, a new InMemoryHistory is created.
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_conversation_chain(vectorstore):
    # conversational retrieval chain using Gemini LLM and LCEL.
    
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        st.error("GOOGLE_API_KEY environment variable is not set. Please set it in your .env file or system environment.")
        return None

    try:
      
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            temperature=0.7,
            google_api_key=google_api_key
        )

        # Gemini models often prefer chat history in a specific format (HumanMessage, AIMessage)
        # MessagesPlaceholder handles this automatically.
        question_generator_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a standalone question to fetch relevant documents. Do NOT answer the question, just rephrase it if necessary to be standalone.")
        ])

        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}")
        ])

        # Create a history-aware retriever
        # This chain takes chat history and a new question, and generates a standalone question for retrieval.
        # The output of llm.invoke for ChatGoogleGenerativeAI is an AIMessage, so we need to extract its content.
        history_aware_retriever = (
            question_generator_prompt
            | llm
            | RunnableLambda(lambda msg: msg.content) # Extract content from AIMessage
            | RunnableLambda(lambda x: vectorstore.as_retriever().invoke(x))
        )

        document_chain = create_stuff_documents_chain(llm, answer_prompt)

        retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)

        conversational_rag_chain = RunnableWithMessageHistory(
            retrieval_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        return conversational_rag_chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {e}")
        st.info("Please check your Google API key and Gemini model accessibility.")
        return None


def handle_userinput(user_question):
  
    if st.session_state.conversation:
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.conversation.invoke(
                    {"input": user_question},
                    config={"configurable": {"session_id": "streamlit_chat_session"}}
                )
                ai_response_content = response['answer']

                st.session_state.chat_history.append(HumanMessage(content=user_question))
                st.session_state.chat_history.append(AIMessage(content=ai_response_content))

            except Exception as e:
                print(f"DEBUG: Full error during conversation: {e}")
                import traceback
                traceback.print_exc()

                st.error(f"Error during conversation: {e}")
                st.info("Please check your terminal for the full error message and traceback.")
                return

        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            elif isinstance(message, AIMessage):
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(f"**{message.type.capitalize()}:** {message.content}", unsafe_allow_html=True)
    else:
        st.warning("Please upload and process your documents first!")


def main():

    load_dotenv()
    st.set_page_config(page_title="RAG Chatbot",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("RAG Chatbot :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF document.")
                return

            # Ensure both API keys are available
            hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
            google_api_key = os.getenv("GOOGLE_API_KEY")

            if not hf_api_token:
                st.error("HUGGINGFACEHUB_API_TOKEN environment variable is not set.")
                return
            if not google_api_key:
                st.error("GOOGLE_API_KEY environment variable is not set.")
                return

            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)

                vectorstore = get_vectorstore(text_chunks)
                if vectorstore is None:
                    return

                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
                if st.session_state.conversation is None:
                    return

                st.success("Documents processed successfully! You can now ask questions.")
                st.markdown("<h2 class='title'>Answer Some Technical Questions to Proceed</h2>", unsafe_allow_html=True)


if __name__ == '__main__':
    main()
