# GDG-RAG-Demo

![Python 3.9](https://github.com/jonathanjthomas/GDG-RAG-Demo/actions/workflows/python-3.9.yml/badge.svg)

# Instructions

## Our environment setup

- **OS:** Windows 11
- **Python Version:** Python 3.12.5
- **CPU:** Intel i7-13700H
- **GPU:** RTX 4060 (8 GB VRAM)
- **RAM:** 16 GB DDR5 (5200 MHz) RAM

## How to run the streamlit application

- Make sure you have Python and Git installed on your system
- Clone the repository on to your local machine using Git
  ```shell
  git clone https://github.com/jonathanjthomas/GDG-RAG-Demo.git
  ```
- Set up a virtual environment using the below command
  ```python
  python -m venv venv
  ```
- Activate your virtual environment using
  - Windows
    ```python
    venv\Scripts\Activate
    ```
  - Linux and MacOS
    ```python
    source venv/bin/activate
    ```
- Install all the required libraries and dependencies
  ```python
  pip install -r requirements.txt
  ```

## File Structure

- **Test Folder**

  - The test folder contains:
    - **File Upload Test:** The files here were test files for checking if the file upload functions were working, feel free to run these files to see it for yourself
    - **Ollama Test:** These python files were to check if the connection with the Ollama server was set up, again feel free to run these to troubleshoot any problems you might be facing

- **Code**
  - The folder contains:
    - **Streamlit:** All the files here set up a streamlit UI for interacting with the LLM
      1. **`ContextStreamlit.py`:** sets up the Streamlit UI to show how the system retrieves the context for a given input
      2. **`chatUIStreamlit.py`:** This is the final UI composing of a chat-like UI for interacting with the LLM along with a sidebar where you can upload files
    - **`upload.py`:** Run this first to upload your files using a Tkinter GUI, this shall be used along with `localrag.py` so you can chat with your documents from the terminal
    - **`localrag.py`:** Run this after running `upload.py` to chat with the document you have uploaded

## Reach Out

Have any doubts? Feel free to reach out to us at:

- Aditya S (as2397@hw.ac.uk)
- Jonathan (jjt2002@hw.ac.uk)
