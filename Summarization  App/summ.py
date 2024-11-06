import streamlit as st
import dotenv
import os
import pickle
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain

dotenv.load_dotenv()

st.subheader("Welcome to Summarization Tool")
st.write("Upload PDF")
selection = st.sidebar.file_uploader('Choose .pdf file only!', type='pdf')

if selection:
    with open(selection.name, 'wb') as f:
        f.write(selection.getvalue())
    
    loader = PyPDFLoader(selection.name)
    data = loader.load()
    st.write("File Uploaded!")
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    docs = text_splitter.split_documents(data)
    
    llm = OpenAI(temperature=0, max_tokens=1000)

    file_path = "vectorstore.pkl"
    
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
    else:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)
    
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
    
    query = "Can you describe what this PDF document is about and summarize the document in 5 bullet points?"
    
    result = chain({"question": query}, return_only_outputs=True)
    
    st.write(result["answer"])
