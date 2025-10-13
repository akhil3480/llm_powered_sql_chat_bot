# tools/make_sqlite_lite.py
import pandas as pd
import sqlalchemy as sa

MYSQL_URL  = "mysql+pymysql://bot_user:3480@localhost:3306/olist"
SQLITE_URL = "sqlite:///data/olist.sqlite"

# Keep core analytics tables; skip 'geolocation' (large) to stay <100MB
TABLES_CORE = [
    "customers",
    "orders",
    "order_items",
    "order_payments",
    "order_reviews",
    "products",
    "sellers",
    "product_category_name_translation",
]

# Optionally reduce columns (saves space) — adjust as you like
COLUMNS = {
    "orders": [
        "order_id","customer_id","order_status",
        "order_purchase_timestamp","order_approved_at",
        "order_delivered_carrier_date","order_delivered_customer_date",
        "order_estimated_delivery_date"
    ],
    "order_items": [
        "order_id","product_id","seller_id","price","freight_value"
    ],
    "order_payments": [
        "order_id","payment_type","payment_installments","payment_value"
    ],
    "order_reviews": [
        "review_id","order_id","review_score","review_creation_date","review_answer_timestamp"
    ],
    "customers": [
        "customer_id","customer_city","customer_state","customer_zip_code_prefix"
    ],
    "sellers": [
        "seller_id","seller_city","seller_state","seller_zip_code_prefix"
    ],
    "products": [
        "product_id","product_category_name"
    ],
    "product_category_name_translation": [
        "product_category_name","product_category_name_english"
    ],
}

mysql  = sa.create_engine(MYSQL_URL)
sqlite = sa.create_engine(SQLITE_URL)

with mysql.connect() as src:
    for t in TABLES_CORE:
        print(f"Exporting {t} ...")
        cols = COLUMNS.get(t)
        if cols:
            df = pd.read_sql(f"SELECT {', '.join(cols)} FROM {t}", src)
        else:
            df = pd.read_sql_table(t, src)
        df.to_sql(t, sqlite, if_exists="replace", index=False)

print("✅ done → data/olist.sqlite (lite)")
