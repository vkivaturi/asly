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

chat = ChatOpenAI(api_key=api_key, base_url=endpoint, model=model_name)

# Set maximum token limit for the chat model
chat.max_tokens = 512

# Configure model-specific parameters
chat.model_kwargs = {"top_p": 0.8, "frequency_penalty": 0.0, "presence_penalty": 0.0, "stop":["<|eot_id|>","<|im_start|>","<|im_end|>"]}

# Define a chat prompt template with pre-defined system and human messages
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Helps writing email in Hindi Language.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Chain the prompt and chat objects to form a processing pipeline
chain = prompt | chat

# Invoke the processing chain with specific messages
response = chain.invoke(
    {
        "messages": [
            HumanMessage(
                content="Please write email in proper format sender name is Virat Kohli and receipent name is Dhoni, about sad performance of RCB in IPL."
            )
        ],
    }
)

# Print the result of the chat processing
print(response.content)

