# GDG-RAG-Demo

## Supported Python versions

![Python 3.12](https://github.com/jonathanjthomas/GDG-RAG-Demo/actions/workflows/python-3.12.yml/badge.svg)

# Instructions

## Recommended system specifications

- **GPU:** 8 GB VRAM
- **RAM:** 16 GB RAM

## How to run the Streamlit application

- Make sure you have Python and Git installed on your system
- Clone the repository on to your local machine using Git

  ```shell
  git clone https://github.com/jonathanjthomas/GDG-RAG-Demo.git
  ```

- Navigate to the repository directory
- Set up a virtual environment using the below command

  ```shell
  python -m venv venv
  ```

- Activate your virtual environment using

  - Windows
    ```shell
    venv\Scripts\Activate
    ```
  - Linux and MacOS
    ```shell
    source venv/bin/activate
    ```

- Install all the required libraries and dependencies

  ```shell
  pip install -r requirements.txt
  ```

- Download and install ![Ollama](https://ollama.com/download)
- Pull the required Ollama models (gemma2:2b and nomic-embed-text)

  ```shell
  ollama pull gemma2:2b
  ollama pull nomic-embed-text
  ```

- Run vault.py with streamlit

  `streamlit run code\vault.py` or `python -m streamlit run code\vault.py`

## Reach Out

Have any doubts? Feel free to reach out to us at:

- Aditya S (as2397@hw.ac.uk)
- Jonathan John Thomas (jjt2002@hw.ac.uk)
