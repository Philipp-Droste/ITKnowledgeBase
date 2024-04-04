import csv
import os
import logging

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
    
    index = 0
    for row in csv_reader:
        if index < 4:
            jobpost_str = row['jobpost']
            jobpost_strs.append(jobpost_str)
        index += 1

logging.info("finished file appending")
jobpost_combined_str = "".join(jobpost_strs)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, length_function=len
)

chunks = text_splitter.split_text(text=jobpost_combined_str)

logging.info("finished splitting")

# Store the chunks part in db (vector)
vectorstore = Neo4jVector.from_texts(
    chunks,
    url=url,
    username=username,
    password=password,
    embedding=embeddings,
    index_name="itkb",
    node_label="JobPosting",
    pre_delete_collection=True,  # Delete existing PDF data
)

logging.info("finished index creation")

qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
)

try:
    result = ""

    if not payload.neo4j_schema:
        extractor = DataExtractor(llm=llm)
        result = extractor.run(data=payload.input)
    else:
        extractor = DataExtractorWithSchema(llm=llm)
        result = extractor.run(schema=payload.neo4j_schema, data=payload.input)

    print("Extracted result: " + str(result))

    disambiguation = DataDisambiguation(llm=llm)
    disambiguation_result = disambiguation.run(result)

    print("Disambiguation result " + str(disambiguation_result))

    return {"data": disambiguation_result}

except Exception as e:
    print(e)
    return f"Error: {e}"

def main():
    query = st.text_input("Ask questions to the knowledge graph")
    logging.info("finished QA display")

    if query:
        stream_handler = StreamHandler(st.empty())
        qa.run(query, callbacks=[stream_handler])

if __name__ == "__main__":
    main()