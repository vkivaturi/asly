# Sample code to create embeddings from a pdf file, load to in memory FAISS vector, query it with a prompt
# and run the prompt + context on LLM

# Importing OpenAI's chat capabilities from langchain_openai
from langchain_openai import ChatOpenAI  
# Importing the function to load environment variables
from dotenv import load_dotenv  

# Importing template and placeholder classes for chat prompts
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
import chromadb

import os  

#Initialise
print("# Initialise") 
 
load_dotenv()
# Retrieve the API key
api_key = os.getenv('API_KEY')  
endpoint = os.getenv('BASE_URL')  
model_name=os.getenv('MODEL')

vectordb_url = os.getenv('VECTOR_URL')  
# chroma_client = chromadb.HttpClient(host=vectordb_url)

print("# Embedding function") 

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

print("# Connection to LLM") 

chat = ChatOpenAI(api_key=api_key, base_url=endpoint, model=model_name)
# Set maximum token limit for the chat model
chat.max_tokens = 512
# Configure model-specific parameters
chat.model_kwargs = {"top_p": 0.8, "frequency_penalty": 0.0, "presence_penalty": 0.0, "stop":["<|eot_id|>","<|im_start|>","<|im_end|>"]}

print("# Create prompt") 

# prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

# <context>
# {context}
# </context>

# Question: {input}""")

prompt = ChatPromptTemplate.from_template("""Answer the following question:
Question: {input}""")

#Load a pdf doc
#loader = PyPDFLoader("C:\\Users\\vijay\\Downloads\\2024ITC_RT14.pdf")
#pages = loader.load_and_split()

#Split doc and load to local vector store
#text_splitter = RecursiveCharacterTextSplitter()
#documents = text_splitter.split_documents(pages)
#vector = FAISS.from_documents(documents, embeddings)

print("# Create vector db client") 

# vectordb = Chroma(
#     client=chroma_client,
#     collection_name="tg_go_embeddings",
#     embedding_function=embedding_function
# )
vectordb = Chroma(persist_directory="/home/vijay/tggo_db", embedding_function=embedding_function)

print("# Create document chain") 

#document_chain = create_stuff_documents_chain(chat, prompt)

print("# Create retriever") 

retriever = vectordb.as_retriever()

print("# Invoke LLM") 

docs = retriever.invoke("What is the amount released for infrastructure facilities?")

#print(docs)

#retrieval_chain = create_retrieval_chain(retriever, document_chain)

#response = retrieval_chain.invoke({"input": "What is the amount released for infrastructure facilities"})

print("# Print response")

print(response["answer"])