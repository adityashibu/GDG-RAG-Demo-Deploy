import streamlit as st
from streamlit_chat import message
import ollama

# Set up Streamlit page configuration
st.set_page_config(page_title="Ollama Chatbot", layout="centered")

# Initialize session state for storing chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""

# Function to handle user input submission
def handle_input():
    user_input = st.session_state["user_input"]
    if user_input:
        # Append user message to the chat history
        st.session_state["messages"].append({"role": "user", "content": user_input})

        # Get the response from Ollama model
        response = ollama.chat(model="gemma2", messages=st.session_state["messages"])
        assistant_response = response['message']['content']

        # Append assistant message to the chat history
        st.session_state["messages"].append({"role": "assistant", "content": assistant_response})

        # Clear the user input field
        st.session_state["user_input"] = ""

# Display chat history
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        message(msg["content"], is_user=True)
    else:
        message(msg["content"])

# Input box with `on_change` callback
st.text_input("Type your message:", key="user_input", on_change=handle_input)

# Add a clear button to reset the chat
if st.button("Clear Chat"):
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]
