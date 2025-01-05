import ollama

# Create a connection to the Ollama model and provide messages in the correct format
response = ollama.chat(model="gemma2", messages=[{"role": "user", "content": "Hello, how are you?"}])

# Print the model's response
print(response['message']['content'])