import json
import os
import sys
import boto3
import streamlit as st

## titan embeddings
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

## data intake
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader

# vector embedding, vector store
from langchain.vectorstores import FAISS

## Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

## bedrock clients
bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)


## data ingestion
def data_ingestion():

    # navigate to folder storing data
    loader=PyPDFDirectoryLoader("data")
    documents=loader.load()

    # - define a text splitter (after testing for optimal approach)
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,
                                                 chunk_overlap=1000)
    
    docs=text_splitter.split_documents(documents)
    return docs

## vector embedding, vector store
def get_vector_store(docs):

    ## declare store
    vectorstore_faiss=FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

# claude model
def get_claude_llm():
    ##create the anthropic model
    llm=Bedrock(model_id="ai21.j2-mid-v1",client=bedrock,
                model_kwargs={'maxTokens':512})
    
    return llm

# llama2 model
def get_llama2_llm():
    ##create the meta model
    llm=Bedrock(model_id="meta.llama2-70b-chat-v1",client=bedrock,
                model_kwargs={'max_gen_len':512})
    
    return llm

# declare a prompt template
# def overall_context():
prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to my final question. Give explanations for your response. If you don't know the answer, 
just say that you don't know, don't make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# get response from model
def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":query})
    return answer['result']


# our attempt at a streamlit "UI"
def main():
    st.set_page_config("Chat w/ PDF")
    
    st.header("Chat with claude and/or llama2")

    user_question = st.text_input("How can I help you?")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)
            llm=get_claude_llm()
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

    if st.button("Llama2 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)
            llm=get_llama2_llm()
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

if __name__ == "__main__":
    main()














