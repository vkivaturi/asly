# Importing OpenAI's chat capabilities from langchain_openai
from langchain_openai import ChatOpenAI  

# Source code reference - https://github.com/ola-krutrim/ai-cloud/blob/main/samples/LangChain/krutrim_langchain.ipynb

# Importing the function to load environment variables
from dotenv import load_dotenv  

# Importing message classes for AI and human interactions
from langchain_core.messages import HumanMessage  

# Importing template and placeholder classes for chat prompts
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  

# Importing the os module for interacting with the operating system
import os  

# Importing json for parsing JSON strings
import json

load_dotenv()

# Retrieve the API key
api_key = os.getenv('API_KEY')  

# Retrieve the base URL
endpoint = os.getenv('BASE_URL')  

# Retrieve the Model
model_name=os.getenv('MODEL')

