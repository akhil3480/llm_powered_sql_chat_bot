# 🤖 LLM-Powered SQL Chat Bot  

A Streamlit application that uses **Hugging Face LLMs** and **Ollama (LLaMA 3)** to generate SQL queries from natural language and assist with database exploration.  
Fully modular, cloud-deployable, and designed for real-world analytics workflows.

---

## 🚀 Live Demo

🔗 **Streamlit App:** https://llm-powered-sql-chatbot-by-akhil.streamlit.app  
🔗 **GitHub Repo:** https://github.com/akhil3480/llm_powered_sql_chat_bot  

---

## 📌 Project Overview

This project is a multi-page **Streamlit web application** that integrates Large Language Models (LLMs) for:

- Natural-language-to-SQL generation  
- Database schema viewing  
- Ad-hoc querying  
- General LLM text assistant functionality  

The backend uses:

- **Hugging Face models** (via Inference API / endpoint)  
- **Ollama** running **LLama 3 8B** locally  
- A modular Python structure that cleanly separates UI, model logic, and database logic  

The app is fully deployed on **Streamlit Cloud**, automatically built from this GitHub repository.

---

## 🧠 LLM Models Used

### Hugging Face Models

The app supports multiple Hugging Face models, selectable from a dropdown in the UI:

- `meta-llama/Meta-Llama-3-8B-Instruct`  
- `google/gemma-7b-it`  
- `HuggingFaceH4/zephyr-7b-beta`  

### Ollama (Local)

- `llama3` (LLaMA-3-8B-Instruct via Ollama)

The app can route prompts to any of the above models depending on user selection.

---

## 🧱 Key Features

### 🔹 1. LLM-Powered SQL Generator
- Converts human questions into SQL queries  
- Uses selected HF/Ollama model to propose SQL  
- Can be extended to enforce “read-only” mode (blocking DROP/DELETE etc.)

### 🔹 2. Database Explorer
- Connect to any SQL database using SQLAlchemy  
- Explore schemas, tables, and columns  
- View column metadata in a structured table

### 🔹 3. Query Runner
- Run SQL queries directly from the browser  
- View results as a Streamlit dataframe  
- Download result as CSV with one click

### 🔹 4. Multi-Model Support
- Choose between multiple Hugging Face models and a local Ollama model  
- Easy to plug in additional models later  
- Model routing and prompts are kept in a dedicated module

### 🔹 5. Streamlit Cloud Deployment
- Auto-deploys from GitHub on every push  
- Uses `requirements.txt` for environment setup  
- No local server configuration needed for the UI

---

## 🧰 Tech Stack

**Frontend / App Framework**
- Streamlit (multi-page app)

**LLM / AI**
- Hugging Face models:
  - `meta-llama/Meta-Llama-3-8B-Instruct`
  - `google/gemma-7b-it`
  - `HuggingFaceH4/zephyr-7b-beta`
- Ollama:
  - `llama3` (LLaMA-3-8B-Instruct)

**Backend / Data**
- Python  
- SQLAlchemy  
- Pandas  

**Deployment**
- GitHub  
- Streamlit Cloud  

---

## ⚙️ Installation (Local)

### 1. Clone the repo

git clone https://github.com/akhil3480/llm_powered_sql_chat_bot
cd llm_powered_sql_chat_bot

### 2. Create and activate a virtual environment (recommended)

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

### 3. Install dependencies

pip install -r requirements.txt

### 4. (Optional) Configure `.env`

DATABASE_URL=postgresql://user:pass@host:port/dbname
HF_API_KEY=your_huggingface_key_here
OLLAMA_BASE_URL=http://localhost:11434

### 5. Run the app locally

streamlit run unified_app.py

---

## 💼 Author

**Akhil Meleth**  
Data & Analytics | LLM & SQL Enthusiast  

