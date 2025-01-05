import torch
import ollama
import os
from openai import OpenAI
import argparse
import json

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Function to open a file an return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()
    
# Function to get the relevant context from the vault based on the user input
