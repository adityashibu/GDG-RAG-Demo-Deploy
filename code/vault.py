import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import tempfile
from uuid import uuid4
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader, JSONLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# ---------------------------- 1 - Data Ingestion ----------------------------

# Function to load the file, split it into chunks, and add them to the vector store
def add_to_vector_store(file, vector_store, chunk_size=1000, chunk_overlap=200):
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
                                                  chunk_overlap=chunk_overlap,
                                                  add_start_index=True,  # track index in original document
                                                )
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

# Function to rewrite the user's query based on recent conversation history
def rewrite_query(user_query, llm, conversation_history):

    # Get the last two messages from the conversation history
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-2:]])

    prompt = ChatPromptTemplate.from_messages(
        [("system","You are a helpful assistant that rewrites user query."),
        ("human", """Rewrite the following user query by incorporating relevant context from the last two messages of the conversation history.
The rewritten query should:

- Preserve the core intent and meaning of the original query
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

Rewritten query:
"""
            ),
        ])

    chain = prompt | llm
    ai_message = chain.invoke(
        {
            "context": context,
            "user_query": user_query,
        }
    )   

    rewritten_query = ai_message.content.strip()

    print("Original query:", user_query)
    print("Rewritten query:", rewritten_query)

    return rewritten_query

# Function to handle the user input submission
def chat(user_query, llm, retriever, conversation_history):   
    # Rewrite the user's query based on the conversation history
    if len(conversation_history) > 1:
        rewritten_query = rewrite_query(user_query, llm, conversation_history)
    else:
        rewritten_query = user_query
        
    # Retrieve relevant context for the rewritten query from the vector database
    retrieved_documents = retriever.invoke(rewritten_query)

    print("Number of retrieved documents:", len(retrieved_documents))

    # Extract the text content of the retrieved documents
    context = "\n\n".join([doc.page_content for doc in retrieved_documents])

    print("\n Retrieved context: ```", context, "```")

    # Create a list of LangChain messages from the conversation history (limit to last 4 messages - starts with human message, ends with AI message)
    messages = [HumanMessage(msg['content']) if msg['role'] == 'user' else AIMessage(msg['content']) for msg in conversation_history[-4:]]
    
    
    # Add system message and human message 
    messages.insert(0, SystemMessage("Answer the following user query using the retrieved context. Provide a concise and informative answer that directly addresses the user's question. Use a maximum of three sentences to answer the question."))
    messages.append(HumanMessage(f"""Question: 
```
{user_query}
```

Context:
```
{context}
```

Answer:
"""
))  

    print("\nMessages:", messages)

    # Generate the response from the model
    return llm.stream(messages)

# ---------------------------- Initialization ----------------------------
print("Initializing...")

# Initialize session state for uploaded files, model, top_k and messages
if 'uploaded_files' not in st.session_state:
    st.session_state['uploaded_files'] = []
if 'model' not in st.session_state:
    st.session_state['model'] = "gemma2:2b"
if 'top_k' not in st.session_state:
    st.session_state['top_k'] = 3  
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

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

# Use the vector store as a retriever
retriever = vector_store.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": st.session_state['top_k']}
)

# ---------------------------- Streamlit UI ----------------------------
# # 1. DISPLAY CHAT MESSAGES
st.title("Vault App")
st.markdown("Welcome to the Vault App! Upload a file and ask a question to retrieve relevant context from the uploaded documents.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. SIDEBAR
# Uploading files
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

# Settings
st.sidebar.header("Settings")
st.session_state["model"] = st.sidebar.selectbox("Select Model", ["gemma2:2b", "gemma2"], index=0) # Model to use
st.session_state["top_k"] = st.sidebar.slider("Top K Context", 1, 5, value=st.session_state.top_k)  # Top K context to retrieve

# Toggle to reset conversation
st.sidebar.button("Reset Conversation", on_click= lambda: st.session_state.update(messages=[]))
    

# 3. USER INPUT
# When the user_query is not None, 
if user_query := st.chat_input("Enter your message"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream = chat(user_query = user_query, 
                    llm = llm, 
                    retriever = retriever,
                    conversation_history = st.session_state['messages'][:-1:])
        
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})