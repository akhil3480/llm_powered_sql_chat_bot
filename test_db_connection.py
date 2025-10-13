import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
engine = create_engine(os.getenv("DB_URL"), pool_pre_ping=True)

with engine.connect() as conn:
    n = conn.execute(text("SELECT COUNT(*) FROM orders")).scalar()

print("Connected successfully! Orders table rows:", n)
