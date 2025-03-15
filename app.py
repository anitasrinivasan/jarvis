import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import Dict, List, Any
import json

# Load environment variables
load_dotenv()

# Initialize OpenAI
def init_openai():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OpenAI API key. Please check your .env file.")
        st.stop()
    return api_key

# Ensure temp directory exists
os.makedirs("temp", exist_ok=True)

# Initialize OpenAI
openai_api_key = init_openai()

# Initialize model
try:
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
except Exception as e:
    st.error(f"Failed to initialize AI model: {str(e)}")
    st.stop()

# Document processing function
def process_document(file_path: str, file_type: str) -> Dict[str, Any]:
    """Process a document and extract its content"""
    try:
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
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        # Store document content in session state
        if 'documents' not in st.session_state:
            st.session_state.documents = []
        
        st.session_state.documents.extend([{
            'content': doc.page_content,
            'metadata': doc.metadata,
            'timestamp': datetime.now().isoformat()
        } for doc in splits])
        
        return {
            'chunk_count': len(splits),
            'processed_file': file_path
        }
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return {'error': str(e)}

def analyze_documents(query: str) -> str:
    """Analyze documents based on a query"""
    if not st.session_state.get('documents'):
        return "No documents have been uploaded yet. Please upload some documents first."
    
    try:
        # Create a prompt template
        prompt = PromptTemplate(
            input_variables=["query", "documents"],
            template="""Based on the following documents, please provide a detailed response to this query: {query}

Documents:
{documents}

Please provide a clear and concise response, highlighting the most relevant information from the documents."""
        )
        
        # Create a chain
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Get relevant document contents
        doc_contents = "\n\n".join([doc['content'] for doc in st.session_state.documents[-5:]])
        
        # Run the chain
        response = chain.run(query=query, documents=doc_contents)
        return response
    except Exception as e:
        return f"Error analyzing documents: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="Document Assistant", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .upload-section {
        padding: 2rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #f0f2f6;
    }
    .assistant-message {
        background-color: #e8f0fe;
    }
</style>
""", unsafe_allow_html=True)

st.title("💬 Document Assistant")

# Sidebar with document list
with st.sidebar:
    st.header("📚 Uploaded Documents")
    if st.session_state.get('documents'):
        for idx, doc in enumerate(st.session_state.documents):
            st.write(f"📄 {doc['metadata'].get('source', f'Document {idx + 1}')}")
    else:
        st.write("No documents uploaded yet")

# Main content area
col1, col2 = st.columns([2, 3])

with col1:
    st.header("📤 Upload Documents")
    with st.container(height=300):
        uploaded_file = st.file_uploader("Upload a document", type=['pdf', 'txt', 'md'])
        if uploaded_file is not None:
            with st.spinner("Processing document..."):
                # Save the file
                with open(f"temp/{uploaded_file.name}", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process the file
                file_type = uploaded_file.name.split(".")[-1].lower()
                result = process_document(f"temp/{uploaded_file.name}", file_type)
                
                if 'error' not in result:
                    st.success(f"✅ Document processed successfully!")
                    st.write(f"Added {result['chunk_count']} sections to the knowledge base")

with col2:
    st.header("💭 Chat with Your Documents")
    
    # Initialize messages if not exists
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing documents..."):
                response = analyze_documents(prompt)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and OpenAI")
