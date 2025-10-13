from openai import OpenAI

# connect to the local Ollama server (default port 11434)
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

resp = client.chat.completions.create(
    model="llama3",  # must match the model you pulled with ollama
    messages=[
        {"role": "system", "content": "You are a helpful SQL assistant."},
        {"role": "user", "content": "Write a simple SQL query that selects all customers from a table."},
    ],
    temperature=0.1,
    max_tokens=60,
)

print("✅ Ollama local model response:\n")
print(resp.choices[0].message.content)
