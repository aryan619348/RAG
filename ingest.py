import os
from dotenv import load_dotenv
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
import tempfile
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.csv_loader import CSVLoader 
from langchain.document_loaders import PyPDFLoader 



def check_file_type(file_path):
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    # Check if the file is a PDF
    if file_extension == '.pdf':
        return 1
    # Check if the file is a CSV
    if file_extension == '.csv':
        return 2
    

def configure_retriever(uploaded_files):
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        check = check_file_type(file)
        if check ==1:
            loader = PyPDFLoader(file) 
        if check ==2:
            loader = CSVLoader(file)
        docs.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create embeddings and store in vectordb
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(splits, embeddings)

    # Define retriever
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})
    print("embeddings created")
    return retriever

