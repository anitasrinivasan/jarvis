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
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Document processing function
def process_document(file_path, file_type, metadata={}):
    """
    Process a document and add it to the vector store
    
    Args:
        file_path (str): Path to the document
        file_type (str): Type of the document (pdf, txt, md)
        metadata (dict): Additional metadata for the document
    
    Returns:
        dict: Processing result
    """
    # Choose appropriate loader based on file type
    if file_type == 'pdf':
        loader = PyPDFLoader(file_path)
    elif file_type == 'txt':
        loader = TextLoader(file_path)
    elif file_type == 'md':
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    # Load and split document
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    # Add metadata to splits
    for split in splits:
        split.metadata.update(metadata)
    
    # Add to vector store
    vectorstore.add_documents(splits)
    
    return {
        'chunk_count': len(splits),
        'processed_file': file_path
    }

# Search tool
def search_knowledge_base(query):
    """Search for information in the knowledge base"""
    try:
        docs = vectorstore.similarity_search(query, k=5)
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "title": doc.metadata.get("title", "Untitled")
            })
        return results
    except Exception as e:
        # Return error information instead of raising an exception
        return f"Error searching knowledge base: {str(e)}"

# Full document retrieval
def get_document_by_title(title):
    """Retrieve a specific document by title"""
    try:
        response = supabase.table("documents").select("*").ilike("metadata->>title", f"%{title}%").execute()
        return response.data
    except Exception as e:
        # Return error information instead of raising an exception
        return f"Error retrieving document: {str(e)}"

# Define tools
tools = [
    Tool(
        name="Search Knowledge Base",
        func=search_knowledge_base,
        description="Useful for searching through the user's documents and finding relevant information"
    ),
    Tool(
        name="Get Document by Title",
        func=get_document_by_title,
        description="Retrieve a specific document by its title"
    )
]

# Create the agent prompt
agent_prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix="You are an AI assistant helping a user search through their personal knowledge base. Use the available tools to find and provide the most relevant information.",
    input_variables=["input", "agent_scratchpad"]
)

# Create the LLM chain
llm_chain = LLMChain(llm=llm, prompt=agent_prompt)

# Create the agent
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)

# Create the agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True  # Add this parameter to handle parsing errors
)

# Direct function for query processing
def process_query(messages):
    try:
        user_message = messages[-1]["content"]
        # If the knowledge base is empty or there's an error, provide a helpful message
        agent_result = agent_executor.run(input=user_message)
        return {"messages": messages + [{"role": "assistant", "content": agent_result}]}
    except Exception as e:
        # Handle any errors and provide a graceful response
        error_message = f"I encountered an issue while processing your request: {str(e)}"
        if "RateLimitError" in str(e):
            error_message = "I'm currently experiencing high demand and have reached my API rate limit. Please try again in a few minutes."
        elif "APIError" in str(e):
            error_message = "There seems to be an issue with the database connection. The system administrator has been notified."
        
        fallback_response = f"I couldn't find specific information about that in your documents. {error_message}"
        return {"messages": messages + [{"role": "assistant", "content": fallback_response}]}

# Streamlit UI
st.title("Second Brain Assistant")

# File upload section
uploaded_file = st.file_uploader("Upload a document", type=['pdf', 'txt', 'md'])
if uploaded_file is not None:
    # Save the file to temp directory
    with open(f"temp/{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Process the file
    file_type = uploaded_file.name.split(".")[-1].lower()
    with st.spinner(f"Processing {file_type} document..."):
        try:
            result = process_document(
                f"temp/{uploaded_file.name}", 
                file_type,
                {"source": uploaded_file.name, "title": uploaded_file.name}
            )
            st.success(f"Document processed successfully! Added {result['chunk_count']} chunks to your knowledge base.")
        except Exception as e:
            if "RateLimitError" in str(e):
                st.error("Unable to process document: OpenAI API rate limit exceeded. Please try again later.")
            elif "APIError" in str(e):
                st.error("Unable to process document: Database connection issue. Please check your Supabase setup.")
            else:
                st.error(f"Error processing document: {str(e)}")

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
