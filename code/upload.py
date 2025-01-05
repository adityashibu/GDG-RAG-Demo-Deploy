import streamlit as st
import PyPDF2
import re
import json

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

# Streamlit UI
st.title("Vault App: PDF, Text, and JSON File Processor")

# PDF to vault.txt section
st.header("Upload a PDF file to append content to vault.txt")
pdf_file = st.file_uploader("Choose a PDF file", type="pdf")
if pdf_file is not None:
    convert_pdf_to_text(pdf_file)

# Add space
st.markdown("<br><br>", unsafe_allow_html=True)

# Text to vault.txt section
st.header("Upload a Text file to append content to vault.txt")
txt_file = st.file_uploader("Choose a Text file", type="txt")
if txt_file is not None:
    upload_txtfile(txt_file)

# Add space
st.markdown("<br><br>", unsafe_allow_html=True)

# JSON to vault.txt section
st.header("Upload a JSON file to append content to vault.txt")
json_file = st.file_uploader("Choose a JSON file", type="json")
if json_file is not None:
    upload_jsonfile(json_file)
