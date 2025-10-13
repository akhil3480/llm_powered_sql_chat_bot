# --- Chat → SQL (diagnostic + production) ---
import os, time
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

st.set_page_config(page_title="SQL Chatbot (MySQL + HF)", layout="wide")
st.title("🔎 Diagnostic — Chat → SQL")

st.write("Boot 1: Streamlit rendered ✅")

# 1) Load env
st.write("Boot 2: Loading .env…")
try:
    load_dotenv()
    DB_URL   = os.getenv("DB_URL")
    HF_TOKEN = os.getenv("HF_TOKEN")
    HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
    st.success(f".env loaded. Model={HF_MODEL}")
    if not DB_URL:
        st.error("DB_URL missing in .env"); st.stop()
except Exception as e:
    st.exception(e); st.stop()

# 2) Read prompt + schema
st.write("Boot 3: Reading prompt & schema…")
try:
    SYSTEM_RULES = open("prompts/sql_generator.txt", encoding="utf-8").read()
    SCHEMA_MD    = open("schema/schema_md.txt",    encoding="utf-8").read()
    st.success("Prompt & schema loaded")
except Exception as e:
    st.exception(e); st.stop()

# 3) Connect DB
st.write("Boot 4: Connecting DB…")
try:
    from sqlalchemy import create_engine, text
    engine = create_engine(DB_URL, pool_pre_ping=True)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    st.success("DB connected")
except Exception as e:
    st.exception(e); st.stop()

# 4) Init Hugging Face client
st.write("Boot 5: Initializing HF client…")
try:
    from huggingface_hub import InferenceClient
    hf = InferenceClient(model=HF_MODEL, token=HF_TOKEN)
    st.success("HF client ready")
except Exception as e:
    st.exception(e); st.stop()

st.divider()
st.write("✅ All boot steps passed. Enter a question below.")

# ---------- Chat helpers ----------
from sqlalchemy import text as sa_text

def llm_sql(user_q: str, schema_md: str, last_sql=None, last_err=None) -> str:
    """
    Ask the LLM for SQL given the schema and the user's question.
    Uses chat_completion (preferred) then falls back to text_generation.
    """
    # Build system + user prompt
    sys_ = SYSTEM_RULES.replace("{{schema_md}}", schema_md)
    prompt = f"{sys_}\n\nUser question:\n{user_q}\n\n-- Return SQL only."
    if last_err:
        prompt += (
            f"\n\nPrevious SQL:\n```sql\n{last_sql}\n```\n"
            f"Error:\n{last_err}\nFix it and return SQL only."
        )

    # Preferred: chat-style (for Mistral and other instruct/chat models)
    try:
        resp = hf.chat_completion(
            messages=[
                {"role": "system", "content": sys_},
                {"role": "user", "content": prompt},
            ],
            max_tokens=350,
            temperature=0.1,
        )
        out = resp.choices[0].message["content"]
    except Exception as e:
        st.warning(f"chat_completion failed, falling back: {e}")
        # Fallback: text_generation
        try:
            out = hf.text_generation(
                prompt,
                max_new_tokens=300,
                temperature=0.3,
            )
            if isinstance(out, dict) and "generated_text" in out:
                out = out["generated_text"]
        except Exception as e2:
            st.error(f"text_generation also failed: {e2}")
            raise

    # Extract SQL from code fences if present
    text_out = str(out)
    if "```" in text_out:
        parts = text_out.split("```")
        sql = parts[-2]
        if sql.strip().lower().startswith("sql"):
            sql = "\n".join(sql.splitlines()[1:])
        return sql.strip()
    return text_out.strip()

def run_sql(sql: str):
    """
    Read-only guard + clean LIMIT handling + execution timer.
    - Blocks write/DDL keywords
    - Keeps only the first LIMIT if multiple
    - Appends LIMIT 200 when safe/missing
    """
    banned = (" update ", " delete ", " insert ", " drop ", " alter ", " create ")
    low = f" {sql.lower()} "
    if any(k in low for k in banned):
        raise ValueError("Write operations are blocked (read-only).")

    # --- Smart LIMIT cleaner ---
    sql = sql.strip().rstrip(";")
    low = sql.lower()

    # If multiple LIMITs exist, keep only the FIRST one.
    if low.count(" limit ") > 1:
        first_idx = low.find(" limit ")
        # keep everything up to first LIMIT (inclusive) and drop the rest after the number
        head = sql[:first_idx]
        tail = sql[first_idx+7:]  # after " limit "
        # get the first token after LIMIT as the number or expression
        first_token = tail.strip().split()[0]
        sql = f"{head} LIMIT {first_token}"
    # If none exist and no COUNT/GROUP, add LIMIT 200
    elif " limit " not in low and " count(" not in low and " group by " not in low:
        sql = sql + " LIMIT 200"

    # Final semicolon
    if not sql.endswith(";"):
        sql += ";"

    t0 = time.time()
    with engine.connect() as conn:
        df = pd.read_sql(sa_text(sql), conn)
    return df, time.time() - t0

# ---------- UI ----------
q = st.chat_input("Ask about your Olist data…")
if q:
    with st.spinner("Thinking…"):
        last_sql = last_err = None
        for _ in range(3):  # retry with self-heal if first SQL errors
            sql = llm_sql(q, SCHEMA_MD, last_sql, last_err)
            try:
                df, secs = run_sql(sql)
                st.subheader("Generated SQL")
                st.code(sql, language="sql")
                st.success(f"Returned {len(df)} rows in {secs:.2f}s")
                st.dataframe(df.head(500), use_container_width=True)
                if len(df.columns) >= 2:
                    st.bar_chart(df.iloc[:20, :2])
                break
            except Exception as e:
                last_sql, last_err = sql, str(e)
                st.warning(f"Retrying due to: {e}")
        else:
            st.error(f"Failed after retries. Last error: {last_err}")

with st.expander("Connection info", expanded=False):
    st.text(f"DB_URL = {DB_URL}")
    st.text(f"Model  = {HF_MODEL}")
