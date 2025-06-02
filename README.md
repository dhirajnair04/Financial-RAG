# 💰 Financial Regulations Chatbot

A conversational AI system designed to answer questions related to **financial regulations** using **Retrieval-Augmented Generation (RAG)**. Built with `Streamlit`, this chatbot leverages an **LLM (Llama-3.2-3B)** and a **FAISS vector store** to provide accurate, document-grounded responses.

---

## 🚀 Features

- 🔍 Retrieval-Augmented Generation (RAG) using financial document embeddings
- 🧠 Local LLM model powered by `GPT4All` (LLaMA 3.2 3B)
- 📚 Vector search with `FAISS` and `HuggingFace Embeddings`
- 🌐 Web-based chatbot interface built with Streamlit
- 💬 Real-time conversational memory and history tracking
- 🎨 Custom UI styling for clean, dark-themed chat bubbles

---

## 📂 Project Structure
```
📦project-root/
├── 📁 Dataset
│   └── Financial Regulations files
├── 📁 models/
│   └── Llama-3.2-3B-Instruct-Q4_0.gguf
├── 📁 financial_data_index/
│   └── faiss index files
├── 📄 new_app.py
├── 📄 script.py
├── 📄 vectordb.py
├── 📄 .gitignore
└── 📄 README.md
```

---

## 🧠 How It Works

1. **Embeddings**: Financial regulation documents are embedded using `sentence-transformers/all-MiniLM-L6-v2`.
2. **Vector Search**: FAISS retrieves relevant chunks from the indexed financial dataset.
3. **LLM Response**: A locally hosted `GPT4All` model generates the answer based on retrieved context.
4. **User Interface**: Streamlit captures the query and displays responses interactively.

---

## 🛠️ Tech Stack

- `Python 3.9+`
- `Streamlit`
- `LangChain`
- `FAISS`
- `PyMuPDF`
- `sentence-transformers`
- `GPT4All (Llama 3.2 3B)`
- `HuggingFace Embeddings`

---

## 🧾 Installation

```bash
# Clone the repo
git clone https://github.com/dhirajnair04/Financial-RAG.git
cd financial-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Run the App

```
streamlit run new_app.py
```
Make sure the FAISS index (```financial_data_index/```) and LLM model (```models/Llama-3.2-3B-Instruct-Q4_0.gguf```) exist and are correctly configured.

---

## 🧑‍⚖️ Use Cases

- Answering regulatory finance queries
- Training compliance officers or students
- Augmenting financial document search in fintech platforms

---

## 📢 Future Improvements

- Add document upload feature for custom regulations
- Integrate speech-to-text for voice queries
- Replace local LLM with API-based or fine-tuned models

---

## 📜 License

This project is open-source under the MIT License. Feel free to use, share, and build on it!
