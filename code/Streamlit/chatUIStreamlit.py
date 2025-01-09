import streamlit as st
import os
import tempfile
from uuid import uuid4


from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader, JSONLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate




# ---------------------------- 1 - Data Ingestion ----------------------------
# Function to load and chunk file into documents
# Parameters:
# - file: The file to upload
# - collection: The collection to upload the file to
# - chunk_size: The size of each chunk to upload (characters)
# - chunk_overlap: The overlap between each chunk (characters)
def add_to_vector_store(file, vector_store, chunk_size=256, chunk_overlap=100):
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

        # Replace temporary file name with original file name in documents' metadata
        for document in data:
            document.metadata["source"] = file.name

        print(f"Loaded {len(data)} documents from {file.name}")
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
        
        print(f"Chunked {file.name} into {len(chunked_data)} pieces")

        # Upload the chunked data to the ChromaDB collection
        uuids = [file.name + str(uuid4()) for _ in range(len(chunked_data))]
        vector_store.add_documents(documents=chunked_data, ids=uuids)

        print(f"Uploaded {file.name} to ChromaDB")
        
        # Delete the temporary file
        tmp.close()
        os.unlink(tmp_file_path)

    

    
        

# ---------------------------- 2 - Query Processing ----------------------------

# Function to get the relevant context from the vault based on the user input
def get_context(query, vault_embeddings, vault_content, top_k=3):
    if vault_embeddings.nelement() == 0:  # Check if the tensor is empty
        return []

    # Encode the rewritten input
    input_embedding = ollama.embeddings(model='nomic-embed-text', prompt=query)["embedding"]

    # Compute the similarity between input and vault embeddings
    cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), vault_embeddings)
    
    # Adjust top k based to ensure it is not more than number of availabl scores
    top_k = min(top_k, len(cos_scores))

    # Sort the scores and get the top-k indices
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()

    # Get the corresponding context from the vault
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context

# Function to rewrite the user's query based on recent conversation history
def rewrite_query(user_query, llm, conversation_history):

    # Get the last two messages from the conversation history
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-2:]])

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that rewrites user query.",
            ),
            (
                "human", 
             """Rewrite the following user query by incorporating relevant context from the last two messages of the conversation history.
                The rewritten query should:

                - Preserve the core intent and meaning of the original query
                - Expand and clarify the query to make it more specific and informative for retrieving relevant context
                - Avoid introducing new topics or queries that deviate from the original query
                - Be concise and clear, without any unnecessary information or repetition
                - Keep the same tone and style as the original query
                - DONT EVER ANSWER the Original query, but instead focus on rephrasing and expanding it into a new query
                - Return the output as plain text, without any additional formatting

                Return ONLY the rewritten query text, without any additional formatting or explanations.

                Conversation History:
                ```
                {context}
                ```

                Original query: 
                ```
                {user_query}
                ```

                Rewritten query: """
            ),
        ])


    chain = prompt | llm
    ai_msg = chain.invoke(
        {
            "context": context,
            "user_query": user_query,
        }
    )   
    print(ai_msg)

    rewritten_query = ai_msg.content.strip()

    return rewritten_query

# Function to handle the user input submission
def chat(user_query, llm, top_k, conversation_history):
    system_message = "You are a helpful assistant specialized answering user query using external context. Your task is to provide accurate and relevant answers to the user's query based solely on the provided context."
   
    # Rewrite the user's query based on the conversation history
    if len(conversation_history) > 1:
        rewritten_query = rewrite_query(user_query, llm, conversation_history)
    else:
        rewritten_query = user_query

    # # Get the relevant context for the rewritten query, but don't display it
    # relevant_context = get_context(rewritten_query, vault_embeddings, vault_content)

    # user_query_with_context = user_query
    # if relevant_context:
    #     context_str = "\n".join(relevant_context)
    #     user_query_with_context = user_query + "\n\nRelevant Context:\n" + context_str

    # # Update the last user message with the relevant context retrieved 
    # conversation_history[-1]["content"] = user_query_with_context

    # messages = [
    #     {"role": "system", "content": system_message},
    #     *conversation_history
    # ]

    # response = client.chat.completions.create(
    #     model=ollama_model,
    #     messages=messages,
    #     max_tokens=2000,
    # )

    # conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})

    # return response.choices[0].message.content


# ---------------------------- Initialization ----------------------------
print("Initializing...")

# Initialize session state for uploaded files, model and top_k 
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'model' not in st.session_state:
    st.session_state.model = "gemma2:2b"
if 'top_k' not in st.session_state:
    st.session_state.top_k = 3  # Default value

# Initialize Chat Ollama model
llm = ChatOllama(
    model = st.session_state["model"],
    temperature = 0.8
)

# Initialize Ollama embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
)

# Initialize chromadb 
vector_store = Chroma(
    collection_name="vault",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

# Use the vectorstore as a retriever
retriever = vector_store.as_retriever()


# ---------------------------- Streamlit UI ----------------------------
st.title("Vault App")
st.subheader("Ask questions about your documents")


# Sidebar for uploading files
st.sidebar.header("Upload a file")
uploaded_files = st.sidebar.file_uploader("Choose a file", 
                                          type=["pdf", "txt", "json", "md"],
                                          accept_multiple_files=True)

# If files have not been loaded into the ChromaDB collection, load them
if uploaded_files:
    new_files = [file for file in uploaded_files if file not in st.session_state.uploaded_files]
    if new_files:
        for new_file in new_files:
            add_to_vector_store(new_file, vector_store)
        st.session_state.uploaded_files.extend(new_files)

# Sidebar for settings
st.sidebar.header("Settings")
st.session_state["model"] = st.sidebar.selectbox("Select Model", ["gemma2:2b"], index=0) # Model to use
st.session_state["top_k"] = st.sidebar.slider("Top K Context", 1, 5, value=st.session_state.top_k)  # Top K context to retrieve

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
user_query = st.text_input("Enter your query:")

if user_query:
    st.session_state['messages'].append({'role': 'user', 'content': user_query})

    # Rewrite the user's query based on the conversation history
    if len(st.session_state['messages']) > 1:
        rewritten_query = rewrite_query(user_query, llm, st.session_state['messages'])
    else:
        rewritten_query = user_query

    st.write(rewritten_query)
    # Call the chat function with updated settings
    # response = chat(user_query = user_query, 
    #                 llm = llm, 
    #                 top_k = st.session_state["top_k"], 
    #                 conversation_history = st.session_state['messages'])
    
    # st.session_state['messages'].append({'role': 'assistant', 'content': response})
    
    # Show the assistant's response in the chat
    # st.chat_message("assistant").markdown(response)
