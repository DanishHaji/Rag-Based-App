import os
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import streamlit as st
from tempfile import NamedTemporaryFile

# Initialize Groq client
client = Groq(api_key="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")

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file_path):
    pdf_reader = PdfReader(pdf_file_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks
def chunk_text(text, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)

# Function to create embeddings and store them in FAISS
def create_embeddings_and_store(chunks, vector_db=None):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if vector_db is None:
        vector_db = FAISS.from_texts(chunks, embedding=embeddings)
    else:
        vector_db.add_texts(chunks)
    return vector_db

# Function to query the vector database and interact with Groq
def query_vector_db(query, vector_db):
    # Retrieve relevant documents
    docs = vector_db.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # Interact with Groq API
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": f"Use the following context:\n{context}"},
            {"role": "user", "content": query},
        ],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

# Streamlit app
st.title("RAG-Based Application QA")

# Upload PDFs
uploaded_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    vector_db = None  # Initialize an empty vector DB
    for uploaded_file in uploaded_files:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            pdf_path = temp_file.name

        # Extract text
        text = extract_text_from_pdf(pdf_path)
        st.write(f"Text extracted from: {uploaded_file.name}")

        # Chunk text
        chunks = chunk_text(text)
        st.write(f"Text chunked from: {uploaded_file.name}")

        # Generate embeddings and store in FAISS
        vector_db = create_embeddings_and_store(chunks, vector_db=vector_db)
        st.write(f"Embeddings generated and stored for: {uploaded_file.name}")

    # User query input
    user_query = st.text_input("Enter your query:")
    if user_query:
        response = query_vector_db(user_query, vector_db)
        st.write("Response from LLM:")
        st.write(response)