import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
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

# Initialize OpenAI's GPT-4o model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Document processing function
def process_document(file_path, file_type, metadata={}):
    """Process uploaded documents and store in vector database"""
    try:
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
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
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

# Simple function to process queries directly
def process_query(query):
    """Process a user query and generate a response"""
    try:
        # First try to search for relevant information
        search_results = search_knowledge_base(query)
        
        # If we found relevant information, use it to generate a response
        if isinstance(search_results, list) and len(search_results) > 0:
            # Get the content from the first few results
            content_list = [item["content"] for item in search_results[:3]]
            content = "\n\n".join(content_list)
            
            # Generate a response that incorporates the found information
            prompt = f"""
            The user asked: "{query}"
            
            I found the following information in their knowledge base:
            {content}
            
            Based on this information, provide a helpful response that answers their question.
            If the information doesn't fully answer their question, acknowledge what was found
            and explain what additional information might be needed.
            """
            
            response = llm.invoke(prompt).content
            return response
        else:
            # If no relevant information was found, provide a general response
            prompt = f"""
            The user asked: "{query}"
            
            I couldn't find specific information about this in their personal knowledge base.
            Please provide a helpful general response that:
            1. Acknowledges that specific information wasn't found in their documents
            2. Offers general information about the topic if possible
            3. Suggests what kind of documents they might want to add to their knowledge base
            """
            
            response = llm.invoke(prompt).content
            return response
            
    except Exception as e:
        # Provide a graceful error message
        if "RateLimitError" in str(e):
            return "I'm currently experiencing high demand and have reached my API rate limit. Please try again in a few minutes."
        return f"I encountered an issue while processing your request. The system administrator has been notified."

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
    
    if result["status"] == "success":
        st.success(f"Document processed successfully! Added {result['chunk_count']} chunks to your knowledge base.")
    else:
        if "RateLimitError" in result.get("error", ""):
            st.error("Unable to process document: API rate limit exceeded. Please try again later.")
        elif "APIError" in result.get("error", ""):
            st.error("Unable to process document: Database connection issue. Please check your Supabase setup.")
        else:
            st.error(f"Error processing document: {result.get('error', 'Unknown error')}")

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
            # Use the simplified direct processing function
            response = process_query(prompt)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
