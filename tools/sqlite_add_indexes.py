import sqlalchemy as sa

# Connect to the SQLite database file
engine = sa.create_engine("sqlite:///data/olist.sqlite")

# Add indexes to speed up common queries
with engine.begin() as conn:
    conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id);")
    conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_order_items_order ON order_items(order_id);")
    conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_order_items_product ON order_items(product_id);")
    conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_order_items_seller ON order_items(seller_id);")
    conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_customers_city ON customers(customer_city);")
    conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_sellers_city ON sellers(seller_city);")

print("Indexes added successfully.")
