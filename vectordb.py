import os
import fitz  # PyMuPDF
import faiss
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import GPT4All
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import login

#login("Your_HF_Token")
# ----------------- STEP 1: Extract Text from PDFs -----------------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# Load all PDFs from a directory
pdf_directory = "D:\Dhiraj\Mtech_Files\Python_code\LLM\Project - Financial RAG\Dataset"  # Folder where PDFs are stored
all_text = ""

for pdf_file in os.listdir(pdf_directory):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, pdf_file)
        all_text += extract_text_from_pdf(pdf_path) + "\n"

print("✅ Text extracted from all PDFs.")

# ----------------- STEP 2: Chunk the Text -----------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
text_chunks = text_splitter.split_text(all_text)
print(f"✅ {len(text_chunks)} text chunks created.")

# ----------------- STEP 3: Generate Embeddings -----------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_texts(text_chunks, embedding_model)

# Save the vector database locally
vector_db.save_local("financial_data_index")
print("✅ Embeddings stored in FAISS.")
