# tools/make_sqlite_mini.py
# Build a small, relationally-consistent SQLite for cloud demo
import pandas as pd
import sqlalchemy as sa

MYSQL_URL   = "mysql+pymysql://bot_user:3480@localhost:3306/olist"
SQLITE_URL  = "sqlite:///data/olist.sqlite"
N_ORDERS    = 30000        # adjust: 20_000–50_000 usually <100MB
RANDOM_SEED = 42

# Tables we’ll keep (skip geolocation; it’s large & not used in most queries)
TABLES_BASE = ["customers","orders","order_items","order_payments","order_reviews",
               "products","sellers","product_category_name_translation"]

mysql  = sa.create_engine(MYSQL_URL)
sqlite = sa.create_engine(SQLITE_URL)

print("Reading orders list…")
orders_full = pd.read_sql("SELECT * FROM orders", mysql)

# Sample order_ids
sample_orders = orders_full.sample(n=min(N_ORDERS, len(orders_full)),
                                   random_state=RANDOM_SEED)
order_ids = set(sample_orders["order_id"].tolist())

print("Filtering dependent tables…")

# Orders (sampled)
orders = sample_orders.copy()

# Order items/payments/reviews filtered by order_id
order_items   = pd.read_sql("SELECT * FROM order_items",   mysql)
order_items   = order_items[order_items["order_id"].isin(order_ids)]

order_payments = pd.read_sql("SELECT * FROM order_payments", mysql)
order_payments = order_payments[order_payments["order_id"].isin(order_ids)]

order_reviews = pd.read_sql("SELECT * FROM order_reviews", mysql)
order_reviews = order_reviews[order_reviews["order_id"].isin(order_ids)]

# Customers referenced by sampled orders
cust_ids = set(orders["customer_id"].tolist())
customers = pd.read_sql("SELECT * FROM customers", mysql)
customers = customers[customers["customer_id"].isin(cust_ids)]

# Products & sellers referenced by sampled items
prod_ids = set(order_items["product_id"].dropna().unique().tolist())
products = pd.read_sql("SELECT product_id, product_category_name FROM products", mysql)
products = products[products["product_id"].isin(prod_ids)]

seller_ids = set(order_items["seller_id"].dropna().unique().tolist())
sellers = pd.read_sql("SELECT seller_id, seller_city, seller_state, seller_zip_code_prefix FROM sellers", mysql)
sellers = sellers[sellers["seller_id"].isin(seller_ids)]

# Category name translations (small; keep all)
trans = pd.read_sql("SELECT * FROM product_category_name_translation", mysql)

print("Writing to SQLite… (this may take a minute)")
with sqlite.begin() as conn:
    customers.to_sql("customers", conn, if_exists="replace", index=False)
    orders.to_sql("orders", conn, if_exists="replace", index=False)
    order_items.to_sql("order_items", conn, if_exists="replace", index=False)
    order_payments.to_sql("order_payments", conn, if_exists="replace", index=False)
    order_reviews.to_sql("order_reviews", conn, if_exists="replace", index=False)
    products.to_sql("products", conn, if_exists="replace", index=False)
    sellers.to_sql("sellers", conn, if_exists="replace", index=False)
    trans.to_sql("product_category_name_translation", conn, if_exists="replace", index=False)

# Shrink file
with sqlite.begin() as conn:
    conn.exec_driver_sql("VACUUM")

print("✅ done → data/olist.sqlite")
print({t: len(df) for t, df in {
    "customers": customers, "orders": orders, "order_items": order_items,
    "order_payments": order_payments, "order_reviews": order_reviews,
    "products": products, "sellers": sellers,
}.items()})
