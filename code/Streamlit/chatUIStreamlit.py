import torch
import ollama
from openai import OpenAI
import json
import streamlit as st
import chromadb
import os
import re
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader, JSONLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile

# TODO
# 1. Implement data loading, chunking, and uploading to chroma with langchain
# 2. Implement retrieval with langchain
# 3. Allow the user to change the collection they are referring to


# ---------------------------- 1 - Data Ingestion ----------------------------


# Function to load and chunk file into documents
# Parameters:
# - file: The file to upload
# - collection: The collection to upload the file to
# - chunk_size: The size of each chunk to upload (characters)
# - chunk_overlap: The overlap between each chunk (characters)
def load_and_chunk(file, collection, chunk_size=1000, chunk_overlap=100):
    if file:
        # Use tempfile because Langchain Loaders only accept a file_path
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.getvalue())
            tmp_file_path = tmp.name

        # Use Langchain Loaders to load the file into a Document object (which stores page content and metadata)
        if file.type == "application/pdf":
            loader = PyPDFLoader(file_path = tmp_file_path)
        elif file.type == "application/json":
            loader = JSONLoader(file_path = tmp_file_path, jq_schema=".", text_content=False)
        elif file.type == "text/markdown":
            loader = UnstructuredMarkdownLoader(file_path = tmp_file_path)        
        else:
            loader = TextLoader(file_path = tmp_file_path)

        data = loader.load()

        # Use Langchain Text Splitter to chunk the document into smaller pieces
        # From LangChain Docs (https://python.langchain.com/docs/how_to/recursive_text_splitter/):
        # This text splitter is the recommended one for generic text. 
        # It is parameterized by a list of characters. It tries to split on them in order until 
        # the chunks are small enough. The default list is ["\n\n", "\n", " ", ""]. 
        # This has the effect of trying to keep all paragraphs (and then sentences, and then words) 
        # together as long as possible, as those would generically seem to be the strongest semantically 
        # related pieces of text.
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                  chunk_overlap=chunk_overlap)
        chunked_data = splitter.split_documents(data)
        
        # Delete the temporary file
        tmp.close()
        os.unlink(tmp_file_path)

        return chunked_data
    

    
        

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
st.title("Vault App")
st.subheader("Ask questions about your documents")

# Sidebar for customization
# st.sidebar.title("Vault App")

# Sidebar for uploading files
# Initialize session state for uploaded files if not already done
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

st.sidebar.header("Upload a file")
uploaded_files = st.sidebar.file_uploader("Choose a file", 
                                          type=["pdf", "txt", "json", "md"],
                                          accept_multiple_files=True)

# If files have not been loaded into the ChromaDB collection, load them
if uploaded_files:
    new_files = [file for file in uploaded_files if file not in st.session_state.uploaded_files]
    if new_files:
        for new_file in new_files:
            load_and_chunk(new_file, collection)
        st.session_state.uploaded_files.extend(new_files)


# Sidebar customization
st.sidebar.header("Settings")
model_option = st.sidebar.selectbox("Select Model", ["gemma2"], index=0)
top_k = st.sidebar.slider("Top K Context", 1, 5)  # Top K context to retrieve

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

# if user_input:
#     # Call the chat function with updated settings
#     response = gemma_chat(user_input, system_message, vault_embeddings_tensor, vault_content, model_option, st.session_state['messages'])
#     st.session_state['messages'].append({'role': 'user', 'content': user_input})
#     st.session_state['messages'].append({'role': 'assistant', 'content': response})
    
#     # Show the assistant's response in the chat
#     st.chat_message("assistant").markdown(response)
