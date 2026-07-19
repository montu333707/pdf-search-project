import transformers

print("Transformers version:", transformers.__version__)
print("Transformers file:", transformers.__file__)

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
)
import streamlit as st
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
)

from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFacePipeline,
)

from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA

# -----------------------
# Load LLM
# -----------------------

checkpoint = "MBZUAI/LaMini-T5-738M"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    torch_dtype=torch.float32,
)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.3,
)

llm = HuggingFacePipeline(pipeline=pipe)

# -----------------------
# Embeddings
# -----------------------

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="db",
    embedding_function=embeddings,
)

retriever = db.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
)

# -----------------------
# Streamlit
# -----------------------

st.title("📄 Search Your PDF")

question = st.text_input("Ask a question")

if st.button("Search"):

    if question:

        answer = qa.invoke({"query": question})

        st.write(answer["result"])  