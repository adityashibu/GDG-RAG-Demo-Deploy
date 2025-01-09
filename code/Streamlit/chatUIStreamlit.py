import torch
import ollama
import os
from openai import OpenAI
import json
import streamlit as st

import re
import PyPDF2

# Function to convert PDF to text and append to vault.txt
def convert_pdf_to_text(file):
    if file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        text = ''
        
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            if page.extract_text():
                text += page.extract_text() + " "
        
        # Normalize whitespace and clean up text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split text into chunks by sentences, respecting a maximum chunk size
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 < 1000:
                current_chunk += (sentence + " ").strip()
            else:
                chunks.append(current_chunk)
                current_chunk = sentence + " "
        if current_chunk:
            chunks.append(current_chunk)
        
        # Append chunks to vault.txt
        with open("vault.txt", "a", encoding="utf-8") as vault_file:
            for chunk in chunks:
                vault_file.write(chunk.strip() + "\n")
        st.success("PDF content appended to vault.txt with each chunk on a separate line.")

# Function to upload and process a text file
def upload_txtfile(file):
    if file:
        text = file.read().decode("utf-8")  # Decode byte stream to string
        
        # Normalize whitespace and clean up text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split text into chunks by sentences, respecting a maximum chunk size
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 < 1000:
                current_chunk += (sentence + " ").strip()
            else:
                chunks.append(current_chunk)
                current_chunk = sentence + " "
        if current_chunk:
            chunks.append(current_chunk)
        
        # Append chunks to vault.txt
        with open("vault.txt", "a", encoding="utf-8") as vault_file:
            for chunk in chunks:
                vault_file.write(chunk.strip() + "\n")
        st.success("Text file content appended to vault.txt with each chunk on a separate line.")

# Function to upload and process a JSON file
def upload_jsonfile(file):
    if file:
        data = json.load(file)  # Load JSON data from the file
        
        # Flatten the JSON data into a single string
        text = json.dumps(data, ensure_ascii=False)
        
        # Normalize whitespace and clean up text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split text into chunks by sentences, respecting a maximum chunk size
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 < 1000:
                current_chunk += (sentence + " ").strip()
            else:
                chunks.append(current_chunk)
                current_chunk = sentence + " "
        if current_chunk:
            chunks.append(current_chunk)
        
        # Append chunks to vault.txt
        with open("vault.txt", "a", encoding="utf-8") as vault_file:
            for chunk in chunks:
                vault_file.write(chunk.strip() + "\n")
        st.success("JSON file content appended to vault.txt with each chunk on a separate line.")

# Function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

# Function to get the relevant context from the vault based on the user input
def get_context(rewritten_input, vault_embeddings, vault_content, top_k=3):
    if vault_embeddings.nelement() == 0:  # Check if the tensor is empty
        return []

    # Encode the rewritten input
    input_embedding = ollama.embeddings(model='mxbai-embed-large', prompt=rewritten_input)["embedding"]
    # Compute the similarity between input and vault embeddings
    cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), vault_embeddings)
    # Adjust top k based on available scores
    top_k = min(top_k, len(cos_scores))

    # Sort the scores and get the top-k indices
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()

    # Get the corresponding context from the vault
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context

def rewrite_query(user_input_json, conversation_history, ollama_model):
    user_input = json.loads(user_input_json)["Query"]
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-2:]])
    prompt = f"""Rewrite the following query by incorporating relevant context from the conversation history.
    The rewritten query should:
    
    - Preserve the core intent and meaning of the original query
    - Expand and clarify the query to make it more specific and informative for retrieving relevant context
    - Avoid introducing new topics or queries that deviate from the original query
    - DONT EVER ANSWER the Original query, but instead focus on rephrasing and expanding it into a new query
    - Return the output as plain text, without any additional formatting
    
    Return ONLY the rewritten query text, without any additional formatting or explanations.
    
    Conversation History:
    {context}
    
    Original query: [{user_input}]
    
    Rewritten query: 
    """
    response = client.chat.completions.create(
        model=ollama_model,
        messages=[{"role": "system", "content": prompt}],
        max_tokens=200,
        n=1,
        temperature=0.1,
    )
    rewritten_query = response.choices[0].message.content.strip()
    return json.dumps({"Rewritten Query": rewritten_query})

# Function to handle the user input submission
def gemma_chat(user_input, system_message, vault_embeddings, vault_content, ollama_model, conversation_history):
    conversation_history.append({"role": "user", "content": user_input})

    if len(conversation_history) > 1:
        query_json = {
            "Query": user_input,
            "Rewritten Query": "",
        }
        rewritten_query_json = rewrite_query(json.dumps(query_json), conversation_history, ollama_model)
        rewritten_query_data = json.loads(rewritten_query_json)
        rewritten_query = rewritten_query_data["Rewritten Query"]
    else:
        rewritten_query = user_input

    # Get the relevant context for the rewritten query, but don't display it
    relevant_context = get_context(rewritten_query, vault_embeddings, vault_content)

    user_input_with_context = user_input
    if relevant_context:
        context_str = "\n".join(relevant_context)
        user_input_with_context = user_input + "\n\nRelevant Context:\n" + context_str

    conversation_history[-1]["content"] = user_input_with_context

    messages = [
        {"role": "system", "content": system_message},
        *conversation_history
    ]

    response = client.chat.completions.create(
        model=ollama_model,
        messages=messages,
        max_tokens=2000,
    )

    conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})

    return response.choices[0].message.content

# Configuration for the Ollama API client
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='llama3'
)

# Load the vault content
vault_content = []
if os.path.exists("vault.txt"):
    with open("vault.txt", "r", encoding='utf-8') as vault_file:
        vault_content = vault_file.readlines()

# Generate embeddings for the vault content using Ollama
vault_embeddings = []
for content in vault_content:
    response = ollama.embeddings(model='mxbai-embed-large', prompt=content)
    vault_embeddings.append(response["embedding"])

# Convert to tensor
vault_embeddings_tensor = torch.tensor(vault_embeddings)

# Streamlit UI
st.title("Document-based Chatbot")
st.subheader("Ask questions about your documents")

# Sidebar for customization
st.sidebar.title("Vault App: PDF, Text, and JSON File Processor")

# PDF to vault.txt section in the sidebar
st.sidebar.header("Upload a PDF file to append content to vault.txt")
pdf_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
if pdf_file is not None:
    convert_pdf_to_text(pdf_file)

# Text to vault.txt section in the sidebar
st.sidebar.header("Upload a Text file to append content to vault.txt")
txt_file = st.sidebar.file_uploader("Choose a Text file", type="txt")
if txt_file is not None:
    upload_txtfile(txt_file)

# JSON to vault.txt section in the sidebar
st.sidebar.header("Upload a JSON file to append content to vault.txt")
json_file = st.sidebar.file_uploader("Choose a JSON file", type="json")
if json_file is not None:
    upload_jsonfile(json_file)

# Sidebar customization
st.sidebar.title("Settings")
model_option = st.sidebar.selectbox("Select Model", ["gemma2", "mxbai-embed-large"], index=0)
top_k = st.sidebar.slider("Top K Context", 1, 5, 3)  # Top K context to retrieve

# Default system message
system_message = "You are a helpful assistant that is an expert at extracting the most useful information from a given text. Also bring in extra relevant information to the user query from outside the given context."

# User input
user_input = st.text_input("Enter your query:")

# Process and display response if the user submits a query
if user_input:
    response = gemma_chat(
        user_input=user_input,
        system_message=system_message,
        vault_embeddings=vault_embeddings_tensor,
        vault_content=vault_content,
        ollama_model=model_option,
        conversation_history=[]
    )
    
    # Display the response with a bot icon
    bot_icon = "https://cdn-icons-png.flaticon.com/512/4712/4712108.png"
    st.markdown("### Response")
    st.markdown(
        f"""
        <div style="display: flex; align-items: flex-start; margin-top: 10px;">
            <div style="width: 40px; height: 40px; background-color: #fff; display: flex; justify-content: center; align-items: center; border-radius: 5px; margin-right: 10px;">
                <img src="{bot_icon}" alt="Bot Icon" style="width: 20px; height: 20px;">
            </div>
            <div style="background-color: #262730; border-radius: 10px; padding: 10px; flex: 1; box-shadow: 0px 2px 5px rgba(0,0,0,0.1);">
                {response}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
