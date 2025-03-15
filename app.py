import os
import streamlit as st
from dotenv import load_dotenv
import logging
import sys
from datetime import datetime
import traceback
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
import json
from tenacity import retry, stop_after_attempt, wait_exponential
import time
from functools import wraps
import threading
from queue import Queue

# Rate limiting setup
class RateLimiter:
    def __init__(self, max_requests, time_window):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = Queue()
        self.lock = threading.Lock()

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.lock:
                current_time = time.time()
                # Remove old requests
                while not self.requests.empty():
                    if current_time - self.requests.queue[0] > self.time_window:
                        self.requests.get()
                    else:
                        break
                
                if self.requests.qsize() >= self.max_requests:
                    sleep_time = self.time_window - (current_time - self.requests.queue[0])
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
                self.requests.put(current_time)
            return func(*args, **kwargs)
        return wrapper

# Retry decorator for API calls
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry_error_callback=lambda retry_state: logger.error(f"Failed after {retry_state.attempt_number} attempts")
)
def api_call_with_retry(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            raise
    return wrapper

# Rate limiters
openai_limiter = RateLimiter(max_requests=50, time_window=60)  # 50 requests per minute
supabase_limiter = RateLimiter(max_requests=100, time_window=60)  # 100 requests per minute

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/app_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Ensure directories exist
os.makedirs("temp", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Load environment variables
load_dotenv()

# Validate required environment variables
required_env_vars = ['OPENAI_API_KEY', 'SUPABASE_URL', 'SUPABASE_SERVICE_KEY']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
    logger.error(error_msg)
    raise RuntimeError(error_msg)

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
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# User Profile Management
def save_user_profile(name: str, interests: List[str], projects: List[str]):
    """Save user profile to Supabase"""
    try:
        profile_data = {
            "name": name,
            "interests": interests,
            "projects": projects
        }
        result = supabase.table("user_profiles").upsert(profile_data).execute()
        return result.data
    except Exception as e:
        st.error(f"Error saving user profile: {str(e)}")
        return None

def get_user_profile():
    """Retrieve user profile from Supabase"""
    try:
        result = supabase.table("user_profiles").select("*").limit(1).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        st.error(f"Error retrieving user profile: {str(e)}")
        return None

def suggest_topics(user_profile: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Generate engaging topics based on user's knowledge base and profile"""
    try:
        # Combine user interests and projects for context
        context = ""
        if user_profile:
            interests_str = ", ".join(user_profile.get("interests", []))
            projects_str = ", ".join(user_profile.get("projects", []))
            context = f"Consider that the user is interested in {interests_str} and working on projects related to {projects_str}. "

        # Search for relevant documents
        query = context + "Find diverse and interesting topics from the knowledge base"
        docs = vectorstore.similarity_search(query, k=5)
        
        # Use LLM to generate topic suggestions
        topics_prompt = f"""Based on the following documents and {context}, suggest 3 engaging topics for discussion:
        {[doc.page_content for doc in docs]}
        
        Format each topic as a JSON object with 'title' and 'description' fields.
        Make the topics diverse and intriguing to spark curiosity."""
        
        response = llm.predict(topics_prompt)
        topics = json.loads(response) if isinstance(response, str) else response
        return topics[:3]
    except Exception as e:
        st.error(f"Error suggesting topics: {str(e)}")
        return []

def get_topic_details(topic: str, user_profile: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get detailed information about a specific topic"""
    try:
        context = ""
        if user_profile:
            interests_str = ", ".join(user_profile.get("interests", []))
            context = f"Consider that the user is interested in {interests_str}. "

        query = f"{context}Find detailed information about: {topic}"
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

# Sidebar for user profile
with st.sidebar:
    st.subheader("Your Profile")
    
    # Initialize session state for profile
    if "profile_setup" not in st.session_state:
        st.session_state.profile_setup = False
        
    if not st.session_state.profile_setup:
        with st.form("profile_form"):
            name = st.text_input("Your Name")
            interests = st.text_input("Your Interests (comma-separated)")
            projects = st.text_input("Current Projects (comma-separated)")
            
            if st.form_submit_button("Save Profile"):
                interests_list = [i.strip() for i in interests.split(",") if i.strip()]
                projects_list = [p.strip() for p in projects.split(",") if p.strip()]
                save_user_profile(name, interests_list, projects_list)
                st.session_state.profile_setup = True
                st.rerun()
    
    # Display current profile
    user_profile = get_user_profile()
    if user_profile:
        st.write(f"👋 Welcome, {user_profile['name']}!")
        st.write("Interests:", ", ".join(user_profile['interests']))
        st.write("Projects:", ", ".join(user_profile['projects']))
        if st.button("Edit Profile"):
            st.session_state.profile_setup = False
            st.rerun()

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
    topics = suggest_topics(user_profile)
    st.subheader("Let's explore your knowledge base! 🚀")
    st.write("I've found some interesting topics we could discuss:")
    
    for topic in topics:
        if st.button(f"📚 {topic['title']}", key=topic['title']):
            st.session_state.current_topic = topic['title']
            st.session_state.topic_details = get_topic_details(topic['title'], user_profile)
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
