import snowflake.connector
import os
import pandas as pd

# Establish Snowflake connection
def connect_to_snowflake():
    return snowflake.connector.connect(
        user=os.getenv('SNOWFLAKE_USERNAME'),
        password=os.getenv('SNOWFLAKE_PASSWORD'),
        account=os.getenv('SNOWFLAKE_ACCOUNT'),
        warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
        database=os.getenv('SNOWFLAKE_DATABASE'),
        schema=os.getenv('SNOWFLAKE_SCHEMA'),
    )

def fetch_user_data(user_df: pd.DataFrame, conn) -> pd.DataFrame:
    """
    Fetches order and review data for a list of user IDs from the database.

    Args:
        user_df (pd.DataFrame): A Pandas Series of CUSTOMER_UNIQUE_IDs.
        conn: A database connection object compatible with pd.read_sql.

    Returns:
        pd.DataFrame: Joined data from customers, orders, order_items, and order_reviews.
    """

    if user_df.empty or user_df.shape[1] != 1:
        raise ValueError("Input must be a one-column DataFrame with user IDs.")

    # Extract the Series from the single-column DataFrame
    user_ids = user_df.iloc[:, 0]  # Get the only column
    
    # Ensure all elements are strings and properly quoted
    formatted_ids = ",".join(f"'{str(uid)}'" for uid in user_ids.unique())

    query = f"""
    SELECT
        c.CUSTOMER_UNIQUE_ID,
        o.ORDER_ID,
        o.ORDER_PURCHASE_TIMESTAMP,
        o.ORDER_ESTIMATED_DELIVERY_DATE,
        o.ORDER_DELIVERED_CUSTOMER_DATE,
        oi.PRICE,
        oi.FREIGHT_VALUE,
        r.REVIEW_SCORE
    FROM OLIST.PUBLIC.CUSTOMERS c
    JOIN OLIST.PUBLIC.ORDERS o ON c.CUSTOMER_ID = o.CUSTOMER_ID
    JOIN OLIST.PUBLIC.ORDER_ITEMS oi ON o.ORDER_ID = oi.ORDER_ID
    LEFT JOIN OLIST.PUBLIC.ORDER_REVIEWS r ON o.ORDER_ID = r.ORDER_ID
    WHERE c.CUSTOMER_UNIQUE_ID IN ({formatted_ids})
    """

    return pd.read_sql(query, conn)


# Feature generation (same as training)
def generate_clv_features(df):
    CUTOFF_DATE = pd.to_datetime("2018-09-01")

    df["ORDER_PURCHASE_TIMESTAMP"] = pd.to_datetime(df["ORDER_PURCHASE_TIMESTAMP"])
    df["TOTAL_PRICE"] = df["PRICE"] + df["FREIGHT_VALUE"]

    feature_df = df[df["ORDER_PURCHASE_TIMESTAMP"] < CUTOFF_DATE]
    print("Length of feature dataframe after cutoff: ", len(feature_df))

    rfm_features = (
        feature_df.groupby("CUSTOMER_UNIQUE_ID")
        .agg(
            recency=("ORDER_PURCHASE_TIMESTAMP", lambda x: (CUTOFF_DATE - x.max()).days),
            frequency=("ORDER_ID", "nunique"),
            monetary=("TOTAL_PRICE", "sum"),
            avg_rating=("REVIEW_SCORE", "mean")
        )
        .fillna(0)
        .reset_index()
    )

    return rfm_features


def generate_churn_features(df):
    CUTOFF_DATE = pd.to_datetime("2018-09-01")

    # Convert timestamps
    df["ORDER_PURCHASE_TIMESTAMP"] = pd.to_datetime(df["ORDER_PURCHASE_TIMESTAMP"])
    df["ORDER_ESTIMATED_DELIVERY_DATE"] = pd.to_datetime(df["ORDER_ESTIMATED_DELIVERY_DATE"])
    df["ORDER_DELIVERED_CUSTOMER_DATE"] = pd.to_datetime(df["ORDER_DELIVERED_CUSTOMER_DATE"])

    # Compute total price and shipping delay
    df["TOTAL_PRICE"] = df["PRICE"] + df["FREIGHT_VALUE"]
    df["SHIPPING_DELAY"] = (df["ORDER_DELIVERED_CUSTOMER_DATE"] - df["ORDER_ESTIMATED_DELIVERY_DATE"]).dt.days
    df["SHIPPING_DELAY"] = df["SHIPPING_DELAY"].fillna(0)

    # Filter customers with at least 2 purchases before cutoff
    features_window = df[df["ORDER_PURCHASE_TIMESTAMP"] < CUTOFF_DATE]
    customer_order_counts = features_window.groupby("CUSTOMER_UNIQUE_ID")["ORDER_ID"].nunique()
    eligible_customers = customer_order_counts[customer_order_counts >= 2].index
    df = df[df["CUSTOMER_UNIQUE_ID"].isin(eligible_customers)]
    features_window = df[df["ORDER_PURCHASE_TIMESTAMP"] < CUTOFF_DATE]
    after_cutoff = df[df["ORDER_PURCHASE_TIMESTAMP"] >= CUTOFF_DATE]

    # Last purchase before cutoff
    last_purchase = (
        features_window.groupby("CUSTOMER_UNIQUE_ID")["ORDER_PURCHASE_TIMESTAMP"]
        .max()
        .reset_index()
        .rename(columns={"ORDER_PURCHASE_TIMESTAMP": "last_pre_cutoff_purchase"})
    )
    # Feature engineering
    features = (
        features_window.groupby("CUSTOMER_UNIQUE_ID")
        .agg(
            recency=("ORDER_PURCHASE_TIMESTAMP", lambda x: (CUTOFF_DATE - x.max()).days),
            frequency=("ORDER_ID", "nunique"),
            monetary=("TOTAL_PRICE", "sum"),
            avg_rating=("REVIEW_SCORE", "mean"),
            avg_shipping_delay=("SHIPPING_DELAY", "mean")
        )
        .fillna(0)
        .reset_index()
    )

    return features

