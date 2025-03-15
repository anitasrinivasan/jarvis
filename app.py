import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from supabase import create_client

# Load environment variables
load_dotenv()

# Initialize Supabase
supabase = create_client(
    os.environ.get("SUPABASE_URL"),
    os.environ.get("SUPABASE_SERVICE_KEY")
)

# Initialize OpenAI
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Initialize vector store
vectorstore = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents"
)

def get_suggested_topics():
    """Get suggested topics from the knowledge base"""
    # Search for diverse content
    docs = vectorstore.similarity_search("", k=5)
    
    # Generate topics
    topics_prompt = """Based on the following content, suggest 3 interesting topics for discussion.
    Each topic should be engaging and thought-provoking.
    
    Content:
    {content}
    
    Format each topic as a brief title followed by a one-sentence description.""".format(
        content="\n\n".join(doc.page_content for doc in docs)
    )
    
    response = llm.invoke(topics_prompt)
    return [line.strip() for line in response.content.split("\n") if line.strip()]

def get_topic_details(topic):
    """Get detailed information about a topic"""
    # Search for relevant content
    docs = vectorstore.similarity_search(topic, k=3)
    
    # Generate detailed response
    detail_prompt = """Based on the following content, provide detailed information about the topic: {topic}
    
    Content:
    {content}
    
    Include:
    1. Key points
    2. Interesting facts
    3. Related questions to explore""".format(
        topic=topic,
        content="\n\n".join(doc.page_content for doc in docs)
    )
    
    return llm.invoke(detail_prompt).content

# Streamlit UI with tabs
st.title("Knowledge Base Explorer")

tab1, tab2 = st.tabs(["Chat & Explore", "Upload Documents"])

# Tab 1: Chat & Explore
with tab1:
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        topics = get_suggested_topics()
        welcome_message = """👋 Welcome! I've found some interesting topics we could discuss:

{topics}

Which topic interests you?""".format(topics="\n".join(f"• {topic}" for topic in topics))
        st.session_state.messages.append({"role": "assistant", "content": welcome_message})
        st.session_state.topics = topics

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # If it's the welcome message, show topic buttons
            if message == st.session_state.messages[0]:
                for topic in st.session_state.topics:
                    if st.button(f"📚 {topic}", key=topic):
                        details = get_topic_details(topic)
                        st.session_state.messages.append({"role": "user", "content": f"Tell me about {topic}"})
                        st.session_state.messages.append({"role": "assistant", "content": details})
                        st.rerun()

    # Chat input
    if prompt := st.chat_input("Ask a question or choose a topic above"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get response
        docs = vectorstore.similarity_search(prompt, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        response_prompt = """Based on the following context, provide a helpful response. If the answer isn't in the context, say "I don't have enough information about that."

Context:
{context}

Question: {question}""".format(context=context, question=prompt)
        
        response = llm.invoke(response_prompt)
        st.session_state.messages.append({"role": "assistant", "content": response.content})
        st.rerun()

# Tab 2: Upload Documents
with tab2:
    st.header("Upload Documents")
    uploaded_file = st.file_uploader("Upload a PDF", type=['pdf'])
    if uploaded_file:
        # Save PDF temporarily
        with open(f"temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("Processing PDF..."):
            # Load and split the PDF
            loader = PyPDFLoader("temp.pdf")
            pages = loader.load()
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(pages)
            
            # Add to vector store
            vectorstore.add_documents(splits)
            
            st.success(f"Added {len(splits)} chunks to your knowledge base!")
            
            # Clean up
            os.remove("temp.pdf")
