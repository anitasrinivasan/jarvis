import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import Tool
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain.chains import LLMChain
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from supabase import create_client

# Ensure temp directory exists
os.makedirs("temp", exist_ok=True)

# Load environment variables
load_dotenv()

# Supabase setup
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase = create_client(supabase_url, supabase_key)

# Create embeddings instance
embeddings = OpenAIEmbeddings()

# Create vector store
vectorstore = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents"
)

# Initialize LLM
llm = ChatOpenAI()

# Document processing function
def process_document(file_path, file_type, metadata={}):
    """Process uploaded documents and store in vector database"""
    # Select appropriate loader based on file type
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
    elif file_type == "txt":
        loader = TextLoader(file_path)
    elif file_type == "md":
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    # Load the document
    documents = loader.load()
    
    # Add metadata
    for doc in documents:
        doc.metadata.update(metadata)
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    
    # Add to vector store
    vectorstore.add_documents(chunks)
    
    return {
        "status": "success",
        "document_count": len(documents),
        "chunk_count": len(chunks)
    }

# Search tool
def search_knowledge_base(query):
    """Search for information in the knowledge base"""
    docs = vectorstore.similarity_search(query, k=5)
    results = []
    for doc in docs:
        results.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Unknown"),
            "title": doc.metadata.get("title", "Untitled")
        })
    return results

# Full document retrieval
def get_document_by_title(title):
    """Retrieve a specific document by title"""
    response = supabase.table("documents").select("*").ilike("metadata->>title", f"%{title}%").execute()
    return response.data

# Define tools
tools = [
    Tool(
        name="search_knowledge_base",
        description="Search for information in the user's saved documents, notes, and bookmarks. Input should be a specific question.",
        func=search_knowledge_base
    ),
    Tool(
        name="get_document_by_title",
        description="Retrieve a complete document by its title or partial title match.",
        func=get_document_by_title
    )
]

# Create agent prompt
PREFIX = """You are a personal knowledge assistant with access to the user's knowledge base.
Your job is to help them find, recall, and use information from their saved content.
You have access to the following tools:"""

FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""

SUFFIX = """Begin!

Question: {input}
Thought: """

# Create the agent
prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=PREFIX,
    format_instructions=FORMAT_INSTRUCTIONS,
    suffix=SUFFIX
)

llm_chain = LLMChain(llm=llm, prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True
)

# Direct function for query processing
def process_query(messages):
    user_message = messages[-1]["content"]
    agent_result = agent_executor.run(input=user_message)
    return {"messages": messages + [{"role": "assistant", "content": agent_result}]}

# Streamlit UI
st.title("Second Brain Assistant")

# File upload
uploaded_file = st.file_uploader("Upload a document to your knowledge base", 
                               type=["pdf", "txt", "md"])

if uploaded_file:
    # Save uploaded file temporarily
    with open(f"temp/{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Process the file
    file_type = uploaded_file.name.split(".")[-1].lower()
    with st.spinner(f"Processing {file_type} document..."):
        result = process_document(
            f"temp/{uploaded_file.name}", 
            file_type,
            {"source": uploaded_file.name, "title": uploaded_file.name}
        )
    
    st.success(f"Document processed successfully! Added {result['chunk_count']} chunks to your knowledge base.")

# Chat interface
st.subheader("Ask about your knowledge")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching for: " + prompt):
            # Use the direct function instead of LangGraph
            response = process_query(st.session_state.messages)
            st.markdown(response["messages"][-1]["content"])
    
    st.session_state.messages.append({"role": "assistant", "content": response["messages"][-1]["content"]})
