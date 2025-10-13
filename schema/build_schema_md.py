import os, pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# 1️⃣ Load .env file (contains DB_URL)
load_dotenv()

# 2️⃣ Create DB connection
engine = create_engine(os.getenv("DB_URL"))

# 3️⃣ Query database structure (table + columns)
q = """
SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = DATABASE()
ORDER BY TABLE_NAME, ORDINAL_POSITION;
"""

# 4️⃣ Read into pandas DataFrame
df = pd.read_sql(q, engine)

# 5️⃣ Format: one line per table -> table(col:type, col:type, ...)
lines = []
for tbl, g in df.groupby("TABLE_NAME"):
    cols = ", ".join(f"{r.COLUMN_NAME}:{r.DATA_TYPE}" for _, r in g.iterrows())
    lines.append(f"{tbl}({cols})")

# 6️⃣ Save into schema/schema_md.txt
os.makedirs("schema", exist_ok=True)
open("schema/schema_md.txt", "w", encoding="utf-8").write("\n".join(lines))
print("✅ schema/schema_md.txt created")
