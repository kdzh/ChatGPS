from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, CSVLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

import streamlit as st

SOURCE_DIR = Path('./data/data_processing')
DATA_DIR = Path('./data/processed_csv_files')


@st.cache_resource
def initialize_vector_db(
        chunk_size=1000,
        chunk_overlap=100,
        embedding_model='sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        db_type='Chroma'
):
    loaders = [
        DirectoryLoader(
            path=DATA_DIR,
            glob="./*.csv",
            # loader_cls=CSVLoader(file_path='./data/processed_csv_files/раздел 5.2.csv', csv_args={'sep': ';'}),
            # loader_cls=CSVLoader,
            loader_cls=CSVLoader,
            loader_kwargs={"csv_args": {"delimiter": ";"}},
        ),
        # DirectoryLoader(
        #     path="./data/data_processing/РАЗДЕЛ 6/",
        #     glob="**/*.docx",
        #     loader_cls=UnstructuredWordDocumentLoader,
        #     loader_kwargs={"mode": "elements"}
        # ),
        DirectoryLoader(
            path=DATA_DIR,  # Replace with your actual path
            glob="**/*.txt",  # Recursively includes all .txt files
            loader_cls=TextLoader  # Use TextLoader for plain text files
        )
    ]

    # Load and combine documents from all loaders
    documents = []
    for loader in loaders:
        docs = loader.load()
        documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model
    )

    if db_type == 'Chroma':
        vector_db = Chroma.from_documents(docs, embeddings)
    elif db_type == 'FAISS':
        vector_db = FAISS.from_documents(docs, embeddings)
    else:
        raise KeyError

    st.session_state.vector_db = vector_db

    return vector_db
