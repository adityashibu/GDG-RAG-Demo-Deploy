# GDG-RAG-Demo

RAG demo for GDG Heriot-Watt University, Dubai

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
