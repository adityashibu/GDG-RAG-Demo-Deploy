import streamlit as st
from streamlit_chat import message

st.title("Chatbot Demo")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]

# Input box for user message
user_input = st.text_input("You:", key="input")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Simulate bot response
    st.session_state.messages.append({"role": "assistant", "content": f"You said: {user_input}"})

# Display chat messages
for message_data in st.session_state.messages:
    if message_data["role"] == "assistant":
        message(message_data["content"], is_user=False)
    else:
        message(message_data["content"], is_user=True)
