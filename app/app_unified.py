# app/app_unified.py
# Chat -> SQL (Unified: Hugging Face API + optional local Ollama) + Query History + Schema TXT Upload

import os
import re
import io
import time
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text as sa_text, text
from datetime import datetime

# ---------------------- page setup ----------------------
st.set_page_config(
    page_title="SQL Chatbot by Akhil Meleth",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom header banner
st.markdown(
    """
    <div style="background-color:#002b36;padding:20px;border-radius:8px;margin-bottom:20px">
        <h1 style="color:#fdf6e3;text-align:center;margin:0;">
            🧠 SQL Chatbot
        </h1>
        <p style="color:#93a1a1;text-align:center;font-size:18px;margin-top:5px;">
            Built with 🔫and 🌹 by <b>Akhil Meleth</b>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------- secrets / env -------------------
load_dotenv()  # local .env

def get_secret(key: str, default=None):
    """Prefer Streamlit Cloud secrets if available, otherwise fallback to local .env."""
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)

DB_URL    = get_secret("DB_URL")  # may be None; we fallback to SQLite below
HF_TOKEN  = get_secret("HF_TOKEN")
HF_MODEL  = get_secret("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
LOCAL_RUN = os.getenv("LOCAL_RUN") == "1"  # set LOCAL_RUN=1 in .env to show Ollama locally

# SQLite fallback (for cloud demo and easy local testing)
if not DB_URL:
    DB_URL = "sqlite:///data/olist.sqlite"

DIALECT = "SQLite" if DB_URL.startswith("sqlite") else "MySQL 8.0"

# ---------------------- load bundled prompt & schema ----
def load_bundled_schema_text() -> str:
    return open("schema/schema_md.txt", encoding="utf-8").read()

def load_prompt_template() -> str:
    return open("prompts/sql_generator.txt", encoding="utf-8").read()

# Build the final SYSTEM_RULES from a given schema text
def build_system_rules(schema_text: str, dialect: str) -> str:
    system_rules = load_prompt_template().replace("{{schema_md}}", schema_text)
    system_rules += f"\n\nSQL DIALECT: {dialect}. Only use features supported by {dialect}."
    if dialect == "SQLite":
        system_rules += (
            "\n- Use date()/datetime()/julianday() instead of MySQL DATE_SUB/INTERVAL."
            "\n- Booleans are 0/1."
            "\n- Single LIMIT only; it must be the last clause."
        )
    return system_rules

try:
    BUNDLED_SCHEMA_MD = load_bundled_schema_text()
except Exception as e:
    st.error(f"Failed to load bundled schema: {e}")
    st.stop()

# ---------------------- DB connect ----------------------
try:
    engine = create_engine(DB_URL, pool_pre_ping=True)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    st.success("DB connected. Prompt and schema loaded.")
except Exception as e:
    st.error(f"DB connection failed: {e}")
    st.stop()

# ---------------------- sidebar: backend & schema source ------
st.sidebar.subheader("Backend")

def show_ollama():
    # Only show locally (LOCAL_RUN=1 in your .env)
    return LOCAL_RUN

backend_options = ["Hugging Face (API)"]
if show_ollama():
    backend_options.append("Ollama (Local)")

backend = st.sidebar.selectbox("Choose backend", backend_options, index=0)

# HF (cloud) models
HF_CHOICES = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "google/gemma-7b-it",
    "HuggingFaceH4/zephyr-7b-beta",
]

# Ollama (local) lightweight model
OLLAMA_CHOICES = [
    "qwen2.5:3b-instruct",
]

if backend.startswith("Hugging Face"):
    model = st.sidebar.selectbox(
        "HF model",
        HF_CHOICES,
        index=HF_CHOICES.index(HF_MODEL) if HF_MODEL in HF_CHOICES else 0
    )
    custom_repo = st.sidebar.text_input("...or custom HF repo (optional)", value="")
    ACTIVE_MODEL = (custom_repo.strip() or model).strip()
    st.sidebar.caption(f"Active HF model: {ACTIVE_MODEL}")
else:
    model = st.sidebar.selectbox("Ollama local model", OLLAMA_CHOICES, index=0)
    custom_repo = st.sidebar.text_input("...or custom Ollama model (optional)", value="")
    ACTIVE_MODEL = (custom_repo.strip() or model).strip()
    st.sidebar.caption(f"Active Ollama model: {ACTIVE_MODEL}")
    st.sidebar.write(f"Tip: first run:  ollama pull {ACTIVE_MODEL}")

# -------- Schema Source: bundled or upload TXT ----------
st.sidebar.markdown("---")
st.sidebar.subheader("Schema Source")

schema_choice = st.sidebar.radio(
    "Provide schema by:",
    ["Use bundled schema", "Upload schema text (.txt)"],
    index=0
)

uploaded_schema_text = None
if schema_choice == "Upload schema text (.txt)":
    up = st.sidebar.file_uploader("Upload .txt file (DDL or Markdown)", type=["txt"])
    if up is not None:
        try:
            uploaded_schema_text = up.read().decode("utf-8", errors="ignore")
            st.sidebar.success("Schema text loaded ✅")
            # preview first 30 lines
            preview = "\n".join(uploaded_schema_text.splitlines()[:30])
            st.sidebar.caption("Preview (first lines):")
            st.sidebar.code(preview, language="markdown")
        except Exception as e:
            st.sidebar.error(f"Failed to read file: {e}")

# choose schema text (uploaded takes priority)
SCHEMA_MD = uploaded_schema_text if uploaded_schema_text else BUNDLED_SCHEMA_MD
SYSTEM_RULES = build_system_rules(SCHEMA_MD, DIALECT)

# -------- Query History controls (sidebar) --------------
st.sidebar.markdown("---")
st.sidebar.subheader("Query History")

if "history" not in st.session_state:
    st.session_state["history"] = []  # list of dicts

save_history = st.sidebar.checkbox("Save to history", value=True)
max_history = st.sidebar.number_input("Keep last N entries", min_value=5, max_value=500, value=50, step=5)

def prune_history():
    if len(st.session_state["history"]) > max_history:
        st.session_state["history"] = st.session_state["history"][-max_history:]

def history_df():
    if not st.session_state["history"]:
        return pd.DataFrame(columns=["when", "backend", "model", "question", "sql", "rows", "time_s"])
    return pd.DataFrame(st.session_state["history"])

def history_csv_bytes():
    df = history_df()
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()

col_h1, col_h2 = st.sidebar.columns(2)
with col_h1:
    if st.sidebar.button("Clear history", use_container_width=True):
        st.session_state["history"] = []
with col_h2:
    st.sidebar.download_button(
        "Download CSV",
        data=history_csv_bytes(),
        file_name="query_history.csv",
        mime="text/csv",
        use_container_width=True
    )

# Small history viewer
hist = history_df()
if not hist.empty:
    st.sidebar.caption("Recent queries:")
    # Show last 5 compact items with rerun buttons
    for idx, row in hist.tail(5).iloc[::-1].iterrows():
        label = f"{row['when']} • {row['question'][:35]}…"
        with st.sidebar.expander(label, expanded=False):
            st.sidebar.code(row["sql"], language="sql")
            if st.sidebar.button("Rerun this", key=f"rerun_{idx}"):
                st.session_state["prefill_query"] = row["question"]

# ---------------------- clients (cached) ----------------
@st.cache_resource(show_spinner=False)
def get_hf_client(model_name: str, token: str):
    from huggingface_hub import InferenceClient
    return InferenceClient(model=model_name, token=token)

@st.cache_resource(show_spinner=False)
def get_ollama_client():
    from openai import OpenAI  # OpenAI python lib talking to Ollama local server
    return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

hf_client = None
ollama_client = None
if backend.startswith("Hugging Face"):
    if not HF_TOKEN:
        st.warning("HF_TOKEN missing. Set it in .env for local or in Streamlit 'Secrets' for cloud.")
    else:
        hf_client = get_hf_client(ACTIVE_MODEL, HF_TOKEN)
else:
    ollama_client = get_ollama_client()

# ---------------------- helpers -------------------------
def extract_sql(text_out: str) -> str:
    text_out = str(text_out or "")
    if "```" in text_out:
        parts = text_out.split("```")
        if len(parts) >= 2:
            sql = parts[-2]
            if sql.strip().lower().startswith("sql"):
                sql = "\n".join(sql.splitlines()[1:])
            return sql.strip()
    return text_out.strip()

def llm_sql(user_q: str, last_sql=None, last_err=None, strict=False) -> str:
    strict_rules = (
        "\nSTRICT MODE:\n"
        "- Use ONLY columns from provided schema.\n"
        "- If GROUP BY is used, ALL non-aggregated selected columns must appear in GROUP BY.\n"
        "- ORDER BY must use selected columns or aliases.\n"
        "- Return ONE runnable SQL statement, end with a semicolon.\n"
        "- Single LIMIT only; it must be last.\n"
    ) if strict else ""

    prompt = (
        f"{SYSTEM_RULES}\n{strict_rules}\n"
        f"User question:\n{user_q}\n\n-- Return SQL only."
    )
    if last_err and last_sql:
        prompt += (
            f"\n\nPrevious SQL (had an error):\n```sql\n{last_sql}\n```\n"
            f"DB error:\n{last_err}\n"
            f"Fix the query. Return SQL only.\n"
        )

    if backend.startswith("Hugging Face"):
        out = None
        if hf_client:
            try:
                # Try chat_completion first
                resp = hf_client.chat_completion(
                    messages=[
                        {"role": "system", "content": SYSTEM_RULES},
                        {"role": "user",   "content": prompt},
                    ],
                    max_tokens=350,
                    temperature=0.0,
                )
                out = resp.choices[0].message["content"]
            except Exception as e:
                # Fallback to text_generation for models that do not support chat_completion
                st.info(f"HF chat_completion failed, falling back: {e}")
                out = hf_client.text_generation(prompt, max_new_tokens=320, temperature=0.1)
                if isinstance(out, dict) and "generated_text" in out:
                    out = out["generated_text"]
        return extract_sql(out)

    else:
        # Ollama (local) via OpenAI-compatible API
        resp = ollama_client.chat.completions.create(
            model=ACTIVE_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_RULES},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            max_tokens=320,
        )
        out = resp.choices[0].message.content
        return extract_sql(out)

def run_sql(sql: str):
    """
    Read-only guard + robust LIMIT handling + show exact executed SQL.
    - Blocks write/DDL keywords
    - Keeps only the first LIMIT if multiple
    - Appends LIMIT 200 when missing
    """
    # 1) Block writes
    low_all = f" {sql.lower()} "
    for k in (" update ", " delete ", " insert ", " drop ", " alter ", " create "):
        if k in low_all:
            raise ValueError("Write operations are blocked (read-only).")

    # 2) Normalize SQL and fix LIMIT (keep only the first LIMIT if any)
    s = (sql or "").strip().rstrip(";")
    m = re.search(r'(?i)\blimit\s+(\d+)', s)
    if m:
        s = s[:m.start()].rstrip() + f" LIMIT {m.group(1)}"
    else:
        s = s + " LIMIT 200"
    if not s.endswith(";"):
        s += ";"

    # 3) Show exact SQL to be executed
    st.caption("Executing (normalized):")
    st.code(s, language="sql")

    # 4) Execute
    t0 = time.time()
    with engine.connect() as conn:
        df = pd.read_sql(sa_text(s), conn)
    return df, time.time() - t0, s

def log_history(when:str, backend:str, model:str, question:str, sql:str, rows:int, seconds:float):
    st.session_state["history"].append({
        "when": when,
        "backend": backend,
        "model": model,
        "question": question,
        "sql": sql,
        "rows": rows,
        "time_s": round(seconds, 3),
    })
    prune_history()

# ---------------------- UI ------------------------------
st.caption("Ask about your data. The app generates SQL, runs it, and shows results.")

# Optional prefill from history "Rerun this"
prefill = st.session_state.pop("prefill_query", None)

# Use either positional placeholder OR keyword, not both
q = st.chat_input(placeholder="Ask about your data... e.g., Top 5 cities by total orders", key="chat_input")

if prefill and not q:
    with st.form("prefill_form", clear_on_submit=True):
        st.info("Rerun previous question:")
        new_q = st.text_input("Question", value=prefill)
        submitted = st.form_submit_button("Run")
        if submitted:
            q = new_q

if q:
    with st.spinner("Thinking..."):
        last_sql = last_err = None
        success = False
        normalized_sql = None
        # normal -> repair -> strict -> strict repair
        for attempt, strict in [(1, False), (2, False), (3, True), (4, True)]:
            sql = llm_sql(q, last_sql=last_sql, last_err=last_err, strict=strict)
            try:
                df, secs, normalized_sql = run_sql(sql)
                st.subheader("Generated SQL")
                st.code(sql or "", language="sql")
                st.success(f"Returned {len(df)} rows in {secs:.2f}s (attempt {attempt})")
                st.dataframe(df.head(500), use_container_width=True)
                if len(df.columns) >= 2 and len(df) > 0:
                    st.bar_chart(df.iloc[:20, :2])

                # history
                if save_history:
                    log_history(
                        when=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        backend=backend,
                        model=ACTIVE_MODEL,
                        question=q,
                        sql=normalized_sql or (sql or ""),
                        rows=len(df),
                        seconds=secs,
                    )
                success = True
                break
            except Exception as e:
                last_sql, last_err = sql, str(e)
                st.warning(f"Retrying (attempt {attempt}) due to: {e}")

        if not success:
            st.error(f"Failed after retries. Last error: {last_err}")
            if last_sql:
                st.caption("Last SQL attempt:")
                st.code(last_sql, language="sql")

with st.expander("Connection & Config", expanded=False):
    st.text(f"DB_URL  = {DB_URL}")
    st.text(f"DIALECT = {DIALECT}")
    st.text(f"Backend = {backend}")
    st.text(f"Model   = {ACTIVE_MODEL}")
    st.text(f"Schema  = {'Uploaded .txt' if uploaded_schema_text else 'Bundled file'}")

# ---------------------- footer ----------------------
st.markdown(
    """
    <hr style="margin-top:40px;margin-bottom:10px;border:1px solid #586e75;">
    <div style="text-align:center;color:#586e75;font-size:14px;">
        © 2025 Created by <b>Akhil Meleth</b> | Powered by Streamlit, Hugging Face, and Ollama
    </div>
    """,
    unsafe_allow_html=True
)
