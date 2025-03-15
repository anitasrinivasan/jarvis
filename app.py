import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import Tool
from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain.chains import LLMChain
from typing import Dict, List, Any
from supabase import Client, create_client
import json
import requests

# Load environment variables
load_dotenv()

# Initialize Supabase client
def init_supabase() -> Client:
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
    
    if not supabase_url or not supabase_key:
        st.error("Missing Supabase credentials. Please check your .env file.")
        st.stop()
    
    try:
        # Initialize with just URL and key, no additional options
        supabase: Client = create_client(supabase_url, supabase_key)
        return supabase
    except Exception as e:
        st.error(f"Failed to initialize Supabase: {str(e)}")
        st.stop()

# Initialize OpenAI
def init_openai():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OpenAI API key. Please check your .env file.")
        st.stop()
    return api_key

# Ensure temp directory exists
os.makedirs("temp", exist_ok=True)

# Initialize services
supabase = init_supabase()
openai_api_key = init_openai()

# Initialize models
try:
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
except Exception as e:
    st.error(f"Failed to initialize AI models: {str(e)}")
    st.stop()

# Create vector store
try:
    vectorstore = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name="documents",
        query_name="match_documents"
    )
except Exception as e:
    st.error(f"Failed to initialize vector store: {str(e)}")
    st.stop()

def suggest_topics() -> List[Dict[str, Any]]:
    """Generate engaging topics based on the knowledge base"""
    try:
        # Search for relevant documents
        query = "Find diverse and interesting topics from the knowledge base"
        docs = vectorstore.similarity_search(query, k=5)
        
        # Use LLM to generate topic suggestions
        topics_prompt = f"""Based on the following documents, suggest 3 engaging topics for discussion:
        {[doc.page_content for doc in docs]}
        
        Format each topic as a JSON object with 'title' and 'description' fields.
        Make the topics diverse and intriguing to spark curiosity."""
        
        response = llm.predict(topics_prompt)
        topics = json.loads(response) if isinstance(response, str) else response
        return topics[:3]
    except Exception as e:
        st.error(f"Error suggesting topics: {str(e)}")
        return []

def get_topic_details(topic: str) -> Dict[str, Any]:
    """Get detailed information about a specific topic"""
    try:
        query = f"Find detailed information about: {topic}"
        docs = vectorstore.similarity_search(query, k=3)
        
        detail_prompt = f"""Based on these documents:
        {[doc.page_content for doc in docs]}
        
        Provide a detailed explanation about '{topic}' with the following structure:
        1. Main insights
        2. Key points
        3. Related topics
        4. Questions to explore further
        
        Format the response as a JSON object with these fields."""
        
        response = llm.predict(detail_prompt)
        return json.loads(response) if isinstance(response, str) else response
    except Exception as e:
        st.error(f"Error getting topic details: {str(e)}")
        return {}

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

# Interactive Topic Discussion
if "current_topic" not in st.session_state:
    st.session_state.current_topic = None
    st.session_state.topic_details = None
    st.session_state.messages = []

# Generate topics if not in conversation
if not st.session_state.messages:
    topics = suggest_topics()
    st.subheader("Let's explore your knowledge base! 🚀")
    st.write("I've found some interesting topics we could discuss:")
    
    for topic in topics:
        if st.button(f"📚 {topic['title']}", key=topic['title']):
            st.session_state.current_topic = topic['title']
            st.session_state.topic_details = get_topic_details(topic['title'])
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Let's explore {topic['title']}! {topic['description']}"
            })
            st.rerun()

# Display conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Show topic details if selected
if st.session_state.current_topic and st.session_state.topic_details:
    with st.expander(f"More about {st.session_state.current_topic}", expanded=True):
        details = st.session_state.topic_details
        st.write("### Main Insights")
        st.write(details.get("main_insights", ""))
        st.write("### Key Points")
        for point in details.get("key_points", []):
            st.write(f"- {point}")
        st.write("### Questions to Explore")
        for question in details.get("questions_to_explore", []):
            if st.button(f"🤔 {question}", key=question):
                st.session_state.messages.append({
                    "role": "user",
                    "content": question
                })
                response = process_query(st.session_state.messages)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["messages"][-1]["content"]
                })
                st.rerun()

# Chat input for follow-up questions
if prompt := st.chat_input("Ask a follow-up question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = process_query(st.session_state.messages)
            st.markdown(response["messages"][-1]["content"])
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": response["messages"][-1]["content"]
    })
