import streamlit as st
import time
import base64
from vectors import EmbeddingsManager  
from chatbot import ChatbotManager    
 
import os
from dotenv import load_dotenv, find_dotenv
from langsmith import Client, traceable
from langchain.prompts import PromptTemplate

load_dotenv(find_dotenv())

# Initialize session_state variables
if 'temp_pdf_path' not in st.session_state:
    st.session_state['temp_pdf_path'] = None

if 'pdf_saved' not in st.session_state:
    st.session_state['pdf_saved'] = False

if 'embeddings_saved' not in st.session_state:
    st.session_state['embeddings_saved'] = False

if 'chatbot_manager' not in st.session_state:
    st.session_state['chatbot_manager'] = None

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Set up Streamlit layout
st.set_page_config(
    page_title="Document Buddy App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar with PDF upload and Embedding button
with st.sidebar:
    st.image("logo.png", use_container_width=True)
    st.markdown("### 📚 Your Personal Document Assistant")
    st.markdown("---")
    
    # Step 1: PDF Upload
    st.subheader("Step 1: Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file is not None:
        st.success("📄 File Uploaded Successfully!")
        st.session_state['temp_pdf_path'] = "temp.pdf"
        
        # Save the uploaded file locally
        with open(st.session_state['temp_pdf_path'], "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state['pdf_saved'] = True
    
    # Step 2: Generate Embeddings
    if st.session_state['pdf_saved']:
        st.subheader("Step 2: Generate Embeddings")
        if st.button("Generate Embeddings") and not st.session_state['embeddings_saved']:
            try:
                embeddings_manager = EmbeddingsManager(
                    model_name="BAAI/bge-small-en",
                    device="cpu",
                    encode_kwargs={"normalize_embeddings": True},
                    qdrant_url="Put your Qdrant URL here",
                    collection_name="vector_db_md"
                )
                
                with st.spinner("Generating Embeddings..."):
                    result = embeddings_manager.create_embeddings(st.session_state['temp_pdf_path'])
                    time.sleep(1)
                st.success("✅ Embeddings created successfully!")
                st.session_state['embeddings_saved'] = True
                
                # Initialize ChatbotManager after embeddings
                st.session_state['chatbot_manager'] = ChatbotManager(
                    model_name="BAAI/bge-small-en",
                    device="cpu",
                    encode_kwargs={"normalize_embeddings": True},
                    llm_model="llama3-70b-8192",
                    llm_temperature=0.7,
                    qdrant_url="Put your Qdrant URL here",
                    collection_name="vector_db_md"
                )
            except Exception as e:
                st.error(f"An error occurred: {e}")
  
# Initialize the LangSmith Client
#client = Client()

def format_prompt(user_input):
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant for document-based queries.",
        },
        {
            "role": "user",
            "content": user_input
        }
    ]

@traceable(run_type="llm", name="Llama 3.2 Model", project_name="Tracing My Project")
def invoke_llm(messages):
    if 'chatbot_manager' not in st.session_state or st.session_state['chatbot_manager'] is None:
        raise ValueError("ChatbotManager is not initialized in session state.")
    # Using the get_response method from ChatbotManager
    user_query = messages[-1]["content"]
    response = st.session_state['chatbot_manager'].get_response(user_query)
    return response

def parse_response(response):
    return response

# Main Chat Interface
st.title("🤖 Chatbot Interface (Llama 3.2 RAG)")

if st.session_state['embeddings_saved']:
    # Chat Display
    st.header("Chat")
    for msg in st.session_state['messages']:
        st.chat_message(msg['role']).markdown(msg['content'])

    # User Input for Chat
    if user_input := st.chat_input("Type your message here..."):
        # Display user message
        st.chat_message("user").markdown(user_input)
        st.session_state['messages'].append({"role": "user", "content": user_input})

        # Run the traceable pipeline with each function traced
        with st.spinner("Responding..."):
            try:
                prompt = format_prompt(user_input)
                response = invoke_llm(prompt)
                answer = parse_response(response)
                time.sleep(1)
            except Exception as e:
                answer = f"⚠️ An error occurred: {e}"
        
        # Display assistant's response
        st.chat_message("assistant").markdown(answer)
        st.session_state['messages'].append({"role": "assistant", "content": answer})
else:
    st.info("Please complete Steps 1 and 2 in the sidebar to start chatting.")
