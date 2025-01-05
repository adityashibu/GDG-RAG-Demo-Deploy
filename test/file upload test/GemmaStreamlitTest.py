import streamlit as st
from streamlit_chat import message
import ollama
from PyPDF2 import PdfReader

# Set up Streamlit page configuration
st.set_page_config(page_title="Ollama Chatbot", layout="centered")

# Initialize session state for storing chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""
if "file_content" not in st.session_state:
    st.session_state["file_content"] = ""

# Function to handle user input submission
def handle_input():
    user_input = st.session_state["user_input"]
    if user_input:
        # Append user message to the chat history
        st.session_state["messages"].append({"role": "user", "content": user_input})

        # Include file content if available
        if st.session_state["file_content"]:
            file_context = f"\nContext from uploaded file:\n{st.session_state['file_content']}"
            st.session_state["messages"].append({"role": "system", "content": file_context})

        # Get the response from Ollama model
        response = ollama.chat(model="gemma2", messages=st.session_state["messages"])
        assistant_response = response['message']['content']

        # Append assistant message to the chat history
        st.session_state["messages"].append({"role": "assistant", "content": assistant_response})

        # Clear the user input field
        st.session_state["user_input"] = ""

# Sidebar for file upload and clear chat
with st.sidebar:
    st.header("Utilities")

    # File upload section
    uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf"], help="Upload text or PDF files for context")
    if uploaded_file is not None:
        st.write("**File Details:**")
        st.write(f"Name: {uploaded_file.name}")
        st.write(f"Type: {uploaded_file.type}")
        st.write(f"Size: {uploaded_file.size} bytes")
        
        if uploaded_file.type == "text/plain":
            # Extract text content from text file
            content = uploaded_file.read().decode("utf-8")
            st.text_area("File Content Preview", content[:500], height=200)
            st.session_state["file_content"] = content  # Store file content

        elif uploaded_file.type == "application/pdf":
            # Extract text content from PDF
            pdf_reader = PdfReader(uploaded_file)
            pdf_text = ""
            for page in pdf_reader.pages:
                pdf_text += page.extract_text()
            st.text_area("File Content Preview", pdf_text[:500], height=200)
            st.session_state["file_content"] = pdf_text  # Store file content

        else:
            st.info("Preview for this file type is not supported yet.")

    # Clear chat button
    if st.button("Clear Chat", use_container_width=True):
        st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]
        st.session_state["file_content"] = ""  # Clear the file content

# Display chat history with a unique key for each message
for i, msg in enumerate(st.session_state["messages"]):
    if msg["role"] == "user":
        message(msg["content"], is_user=True, key=f"user_{i}")  # Unique key for user messages
    else:
        message(msg["content"], key=f"assistant_{i}")  # Unique key for assistant messages

# Input box with `on_change` callback
st.text_input("Type your message:", key="user_input", on_change=handle_input)
