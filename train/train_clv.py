import snowflake.connector
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import pandas as pd
import joblib
from dotenv import load_dotenv
import os

load_dotenv()

# Connect to Snowflake
conn = snowflake.connector.connect(
    user=os.getenv('SNOWFLAKE_USERNAME'),
    password=os.getenv('SNOWFLAKE_PASSWORD'),
    account=os.getenv('SNOWFLAKE_ACCOUNT'),
    warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
    database=os.getenv('SNOWFLAKE_DATABASE'),
    schema=os.getenv('SNOWFLAKE_SCHEMA'),
)

CUTOFF_DATE = pd.to_datetime("2018-03-01")  # 6 months before dataset end
END_DATE = pd.to_datetime("2018-09-01")     # dataset ends in 2018-09

def fetch_all_order_data():
    query = """
    SELECT
      c.CUSTOMER_UNIQUE_ID,
      o.ORDER_ID,
      o.ORDER_PURCHASE_TIMESTAMP,
      oi.PRICE,
      oi.FREIGHT_VALUE,
      r.REVIEW_SCORE
    FROM OLIST.PUBLIC.CUSTOMERS c
    JOIN OLIST.PUBLIC.ORDERS o ON c.CUSTOMER_ID = o.CUSTOMER_ID
    JOIN OLIST.PUBLIC.ORDER_ITEMS oi ON o.ORDER_ID = oi.ORDER_ID
    LEFT JOIN OLIST.PUBLIC.ORDER_REVIEWS r ON o.ORDER_ID = r.ORDER_ID
    """
    return pd.read_sql(query, conn)


def generate_features_and_target(df):
    df["ORDER_PURCHASE_TIMESTAMP"] = pd.to_datetime(df["ORDER_PURCHASE_TIMESTAMP"])
    
    # Revenue
    df["TOTAL_PRICE"] = df["PRICE"] + df["FREIGHT_VALUE"]

    # Split into feature and target windows
    feature_df = df[df["ORDER_PURCHASE_TIMESTAMP"] < CUTOFF_DATE]
    target_df = df[(df["ORDER_PURCHASE_TIMESTAMP"] >= CUTOFF_DATE) & (df["ORDER_PURCHASE_TIMESTAMP"] <= END_DATE)]

    # Compute RFM + avg rating from the feature window
    rfm_features = (
        feature_df.groupby("CUSTOMER_UNIQUE_ID")
        .agg(
            recency=("ORDER_PURCHASE_TIMESTAMP", lambda x: (CUTOFF_DATE - x.max()).days),
            frequency=("ORDER_ID", "nunique"),
            monetary=("TOTAL_PRICE", "sum"),
            avg_rating=("REVIEW_SCORE", "mean")
        )
        .fillna(0)
    )

    # Compute total future spend (CLV) from the target window
    target = (
        target_df.groupby("CUSTOMER_UNIQUE_ID")["TOTAL_PRICE"]
        .sum()
        .rename("future_monetary")
    )

    # Merge
    full_data = rfm_features.join(target, how="inner").reset_index()
    return full_data

def train_future_clv_model(df):
    df = df.dropna()

    X = df[["recency", "frequency", "monetary", "avg_rating"]]
    y = df["future_monetary"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    rmse = root_mean_squared_error(y_test, model.predict(X_test))
    print(f"RMSE: {rmse:.2f}")

    feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("Feature importance:\n", feature_importance)

    return model


if __name__ == "__main__":
    raw_df = fetch_all_order_data()
    training_df = generate_features_and_target(raw_df)
    model = train_future_clv_model(training_df)
    joblib.dump(model, "../models/future_clv_model.joblib")
    print("Model saved to future_clv_model.joblib")

