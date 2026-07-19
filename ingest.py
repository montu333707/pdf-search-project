from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
from constants import CHROMA_SETTINGS

persist_directory = 'db'
def main():
    for root ,dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PDFMinerLoader(os.path.join(root ,file))
    documents = loader.load()
    text_splitter =RecursiveCharacterTextSplitter(chunk_size = 500 ,chunk_overlap =500)
    texts = text_splitter.split_documents(documents)
    #create embeddings here
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    #create vector store here 
    db = Chroma.from_documents(documents=texts,embedding=embeddings,persist_directory=persist_directory)
    # db.persist()
    db =None


if __name__ == "__main__":
    main()