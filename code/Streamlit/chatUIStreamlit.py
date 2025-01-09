import torch
import ollama
from openai import OpenAI
import json
import streamlit as st
import chromadb
import re
import PyPDF2

# ---------------------------- 1 - Data Ingestion ----------------------------

# Function to split text into chunks by sentences, respecting a maximum chunk size
def chunk(text, max_length=1000):
    # Normalize whitespace and clean up text
    text = re.sub(r'\s+', ' ', text).strip()

    # Split text into chunks by sentences, respecting a maximum chunk size
    sentences = re.split(r'(?<=[.!?]) +', text)

    # Split the sentences into chunks
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 < max_length:
            current_chunk += (sentence + " ").strip()
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

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
        
        # Split text into chunks by sentences, respecting a maximum chunk size
        chunks = chunk(text)
        
        # Append chunks to vault.txt
        with open("vault.txt", "a", encoding="utf-8") as vault_file:
            for chunk in chunks:
                vault_file.write(chunk.strip() + "\n")
        st.success("PDF content appended to vault.txt with each chunk on a separate line.")

# Function to upload and process a text file
def upload_txtfile(file):
    if file:
        text = file.read().decode("utf-8")  # Decode byte stream to string
        
        # Split text into chunks by sentences, respecting a maximum chunk size
        chunks = chunk(text)
        
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
        
        # Split text into chunks by sentences, respecting a maximum chunk size
        chunks = chunk(text)
        
        # Append chunks to vault.txt
        with open("vault.txt", "a", encoding="utf-8") as vault_file:
            for chunk in chunks:
                vault_file.write(chunk.strip() + "\n")
        st.success("JSON file content appended to vault.txt with each chunk on a separate line.")


# ---------------------------- 2 - Query Processing ----------------------------

# Function to get the relevant context from the vault based on the user input
def get_context(rewritten_input, vault_embeddings, vault_content, top_k=3):
    if vault_embeddings.nelement() == 0:  # Check if the tensor is empty
        return []

    # Encode the rewritten input
    input_embedding = ollama.embeddings(model='nomic-embed-text', prompt=rewritten_input)["embedding"]

    # Compute the similarity between input and vault embeddings
    cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), vault_embeddings)
    
    # Adjust top k based to ensure it is not more than number of availabl scores
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

    # Update the last user message with the relevant context retrieved 
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



# ---------------------------- Initialization ----------------------------

# Configuration for the Ollama API client
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='gemma2'
)

# Initialize the chromadb client and collection
chroma_client = chromadb.PersistentClient(path="chroma-data")
collection = chroma_client.get_or_create_collection(name="vault_collection")



# ---------------------------- Streamlit UI ----------------------------
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
model_option = st.sidebar.selectbox("Select Model", ["gemma2"], index=0)
top_k = st.sidebar.slider("Top K Context", 1, 3, 5)  # Top K context to retrieve

# Default system message
system_message = "You are a helpful assistant that is an expert at extracting the most useful information from a given text. Also bring in extra relevant information to the user query from outside the given context."

# Conversation history
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Display chat history
for message in st.session_state['messages']:
    if message['role'] == 'user':
        st.chat_message("user").markdown(message['content'])
    else:
        st.chat_message("assistant").markdown(message['content'])

# User input
user_input = st.text_input("Enter your query:")

if user_input:
    # Call the chat function with updated settings
    response = gemma_chat(user_input, system_message, vault_embeddings_tensor, vault_content, model_option, st.session_state['messages'])
    st.session_state['messages'].append({'role': 'user', 'content': user_input})
    st.session_state['messages'].append({'role': 'assistant', 'content': response})
    
    # Show the assistant's response in the chat
    st.chat_message("assistant").markdown(response)
