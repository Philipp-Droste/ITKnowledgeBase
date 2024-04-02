import csv
import os

import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Neo4jVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from stream_handler import StreamHandler
from chains import load_llm

url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

llm, embeddings, dimension = load_llm()

file_path = 'job_posts.csv'
jobpost_strs = []
with open(file_path, 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    
    for row in csv_reader:
        jobpost_str = row['jobpost']
        jobpost_strs.append(jobpost_str)
jobpost_combined_str = "".join(jobpost_strs)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, length_function=len
)

chunks = text_splitter.split_text(text=jobpost_combined_str)

# Store the chunks part in db (vector)
vectorstore = Neo4jVector.from_texts(
    chunks,
    url=url,
    username=username,
    password=password,
    embedding=embeddings,
    index_name="pdf_bot",
    node_label="PdfBotChunk",
    pre_delete_collection=True,  # Delete existing PDF data
)

qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
)

def main():
    query = st.text_input("Ask questions to the knowledge graph")

    if query:
        stream_handler = StreamHandler(st.empty())
        qa.run(query, callbacks=[stream_handler])

if __name__ == "__main__":
    main()