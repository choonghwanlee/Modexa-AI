import os
import pandas as pd
import numpy as np
import joblib
import snowflake.connector
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
from dotenv import load_dotenv

load_dotenv()

# Constants
CUTOFF_DATE = pd.to_datetime("2017-09-01")
CHURN_LOOKAHEAD = pd.Timedelta(days=180)
END_DATE = pd.to_datetime("2018-09-01")

# Connect to Snowflake
conn = snowflake.connector.connect(
    user=os.getenv("SNOWFLAKE_USERNAME"),
    password=os.getenv("SNOWFLAKE_PASSWORD"),
    account=os.getenv("SNOWFLAKE_ACCOUNT"),
    warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
    database=os.getenv("SNOWFLAKE_DATABASE"),
    schema=os.getenv("SNOWFLAKE_SCHEMA"),
)

def fetch_all_order_data():
    query = """
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
    """
    return pd.read_sql(query, conn)

def generate_churn_features_and_labels(df):
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

    # Label creation
    label_df = last_purchase.copy()
    label_df["churn"] = 1  # default: churned

    for idx, row in label_df.iterrows():
        cust_id = row["CUSTOMER_UNIQUE_ID"]
        start = row["last_pre_cutoff_purchase"]
        end = start + CHURN_LOOKAHEAD

        if end > END_DATE:
            continue  # skip incomplete windows

        future_orders = after_cutoff[
            (after_cutoff["CUSTOMER_UNIQUE_ID"] == cust_id) &
            (after_cutoff["ORDER_PURCHASE_TIMESTAMP"] > start) &
            (after_cutoff["ORDER_PURCHASE_TIMESTAMP"] <= end)
        ]
        if not future_orders.empty:
            label_df.at[idx, "churn"] = 0

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

    df_final = pd.merge(features, label_df[["CUSTOMER_UNIQUE_ID", "churn"]], on="CUSTOMER_UNIQUE_ID", how="inner")
    return df_final

def train_churn_model(df):
    df = df.dropna()
    print("Original class balance:\n", df["churn"].value_counts())

    X = df[["recency", "frequency", "monetary", "avg_rating", "avg_shipping_delay"]]
    y = df["churn"]

    # Downsample majority class (churned) to match minority class (active)
    df_majority = df[df.churn == 1]
    df_minority = df[df.churn == 0]
    df_majority_downsampled = resample(df_majority, replace=False,
                                       n_samples=len(df_minority),
                                       random_state=42)
    df_balanced = pd.concat([df_majority_downsampled, df_minority])
    print("Balanced class counts:\n", df_balanced["churn"].value_counts())

    X = df_balanced[["recency", "frequency", "monetary", "avg_rating", "avg_shipping_delay"]]
    y = df_balanced["churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("✅ Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return model

if __name__ == "__main__":
    df_orders = fetch_all_order_data()
    df_churn = generate_churn_features_and_labels(df_orders)
    churn_model = train_churn_model(df_churn)
    joblib.dump(churn_model, "../models/churn_model.joblib")

