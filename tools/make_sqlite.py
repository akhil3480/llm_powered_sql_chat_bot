# tools/make_sqlite.py
import pandas as pd
import sqlalchemy as sa

# 1) CONNECT: MySQL (source) 
MYSQL_URL  = "mysql+pymysql://bot_user:3480@localhost:3306/olist"

# 2) CONNECT: SQLite (target in repo)
SQLITE_URL = "sqlite:///data/olist.sqlite"

# 3) TABLES to export (you can trim later if file is too big for GitHub)
TABLES = [
    "customers",
    "orders",
    "order_items",
    "order_payments",
    "order_reviews",
    "products",
    "sellers",
    "geolocation",
    "product_category_name_translation",
]

mysql_engine  = sa.create_engine(MYSQL_URL)
sqlite_engine = sa.create_engine(SQLITE_URL)

with mysql_engine.connect() as src:
    for t in TABLES:
        print(f"Exporting {t} ...")
        df = pd.read_sql_table(t, src)
        df.to_sql(t, sqlite_engine, if_exists="replace", index=False)

print("✅ done → data/olist.sqlite")
