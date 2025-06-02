import os
import fitz  # PyMuPDF
import faiss
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import GPT4All

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Financial AI Chatbot", page_icon="💰", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #f5f7fa;
        }
        .stTextInput>div>div>input {
            font-size: 18px;
            padding: 10px;
        }
        .chat-container {
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
        }
        .user-msg {
            background-color: #1e293b;
            color: #ffffff;
            text-align: left;
            border-radius: 10px;
            padding: 10px;
        }
        .bot-msg {
            background-color: #1e293b;
            color: #ffffff;
            border-radius: 10px;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("💰 Financial Regulations Chatbot")

# ----------------- Load the RAG System -----------------
@st.cache_resource
def load_rag_system():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.load_local("financial_data_index", embedding_model, allow_dangerous_deserialization=True)
    retriever = vector_db.as_retriever()
    llm = GPT4All(model="D:/Dhiraj/Mtech_Files/Python_code/LLM/Project - Financial RAG/models/Llama-3.2-3B-Instruct-Q4_0.gguf", device="cpu", verbose=True)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

qa_chain = load_rag_system()

# ----------------- Chat Functionality -----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.text_input("Ask a financial regulation question:")

if st.button("Send"):
    if user_query:
        # Append user message
        st.session_state.chat_history.append(("You", user_query))
        
        # Get bot response
        bot_response = qa_chain.run(user_query)
        
        # Append bot message
        st.session_state.chat_history.append(("AI Bot", bot_response))

# Display chat history
for sender, msg in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f'<div class="chat-container user-msg">{msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-container bot-msg">{msg}</div>', unsafe_allow_html=True)