
# It contains all the logic and calculations behind the data integration.


# Libraries
import pandas as pd



# PART 1 : Data Upload

# Loading Data
def load_data(trans_file, cust_file):
    df_Transaction = pd.read_csv(trans_file)
    df_CustMaster = pd.read_csv(cust_file, encoding="latin1")
    return df_Transaction, df_CustMaster

# Cleaning Data
def clean_data(df):
    df.columns = df.columns.str.strip().str.upper()

    if "CUSTOMER_ID" in df.columns:
        df["CUSTOMER_ID"] = df["CUSTOMER_ID"].astype(str).str.strip()

    return df

# Merging Data
def merge_data(df_T, df_C):
    df_T.columns = df_T.columns.str.upper()
    df_C.columns = df_C.columns.str.upper()

    key_T = [col for col in df_T.columns if "CUST" in col or "ACCOUNT" in col][0]
    key_C = [col for col in df_C.columns if "CUST" in col or "ACCOUNT" in col][0]

    df_T["CUSTOMER_ID"] = df_T[key_T]
    df_C["CUSTOMER_ID"] = df_C[key_C]

    df = pd.merge(df_T, df_C, on="CUSTOMER_ID", how="inner")
    return df



# PART 2 : Recommendation System Build

import pickle
import os

required_files = [
    "upsellmodel.pkl",
    "upsellmodel_columns.pkl",
    "prediction_threshold.pkl"
]

for file in required_files:
    if not os.path.exists(file):
        raise FileNotFoundError(f"{file} not found")


# Build similarity matrix from uploaded data
def build_similarity_from_uploaded_data(df):

    from sklearn.preprocessing import MinMaxScaler, normalize
    from sklearn.metrics.pairwise import cosine_similarity
    import pandas as pd

    df = df.copy()

    df.columns = df.columns.str.strip().str.upper()

    df["BILL_DATE"] = pd.to_datetime(
        df["BILL_DATE"],
        dayfirst=True,
        errors="coerce"
    )

    df = df.dropna(subset=["BILL_DATE"])

    df["QTY"] = pd.to_numeric(
        df["QTY"],
        errors="coerce"
    ).fillna(0)

    df["NSV"] = pd.to_numeric(
        df["NSV"],
        errors="coerce"
    ).fillna(0)

    df = df[df["QTY"] > 0]

    df["Recency"] = (
        df["BILL_DATE"].max()
        - df["BILL_DATE"]
    ).dt.days

    df["recency_weight"] = (
        1 / (df["Recency"] + 1)
    )

    df["value_weight"] = (
        df["NSV"]
        /
        (df["QTY"] + 1)
    )

    scaler = MinMaxScaler()

    df["value_weight"] = scaler.fit_transform(
        df[["value_weight"]]
    )

    customer_product_freq = (
        df.groupby(
            ["CUSTOMER_ID", "PRODUCT_DESCRIPTION"]
        )
        .size()
        .reset_index(name="purchase_frequency")
    )

    df = df.merge(
        customer_product_freq,
        on=["CUSTOMER_ID", "PRODUCT_DESCRIPTION"],
        how="left"
    )

    df["frequency_weight"] = (
        df["purchase_frequency"]
        /
        df["purchase_frequency"].max()
    )

    df["final_weight"] = (
        0.4 * df["recency_weight"]
        +
        0.3 * df["value_weight"]
        +
        0.3 * df["frequency_weight"]
    )

    # -----------------------------------
    # LIMIT PRODUCT UNIVERSE FOR SPEED
    # -----------------------------------

    TOP_PRODUCT_LIMIT = 300

    top_product_list = (
        df["PRODUCT_DESCRIPTION"]
        .value_counts()
        .head(TOP_PRODUCT_LIMIT)
        .index
    )

    df = df[
        df["PRODUCT_DESCRIPTION"].isin(top_product_list)
    ]

    basket = pd.pivot_table(
        df,
        index="CUSTOMER_ID",
        columns="PRODUCT_DESCRIPTION",
        values="final_weight",
        aggfunc="sum"
    ).fillna(0)

    basket_normalized = normalize(
        basket,
        norm="l2"
    )

    basket_normalized = pd.DataFrame(
        basket_normalized,
        index=basket.index,
        columns=basket.columns
    )

    similarity = cosine_similarity(
        basket_normalized.T
    )

    sim_df = pd.DataFrame(
        similarity,
        index=basket_normalized.columns,
        columns=basket_normalized.columns
    )

    top_products = (
        df["PRODUCT_DESCRIPTION"]
        .value_counts()
        .index
        .tolist()
    )

    return sim_df, top_products


# Recommendation Function
def recommend_for_customer(customer_id, purchased, sim_df, top_products, upsell_prob):

    purchased = [
        str(p).strip()
        for p in purchased
        if str(p).strip() != ""
    ]

    recs = {}

    for p in purchased:

        if p in sim_df.columns:

            similar = sim_df[p].sort_values(
                ascending=False
            )[1:10]

            for item, score in similar.items():

                if item not in purchased:

                    recs[item] = recs.get(item, 0) + score

    recs_sorted = sorted(
        recs.items(),
        key=lambda x: x[1],
        reverse=True
    )

    recs_sorted = [x[0] for x in recs_sorted]

    if upsell_prob >= 0.6:
        rec_count = 3
    else:
        rec_count = 2

    final_recs = recs_sorted[:rec_count]

    # Fallback but NEVER recommend already purchased products
    if len(final_recs) < rec_count:

        fallback_recs = [
            item for item in top_products
            if item not in purchased and item not in final_recs
        ]

        final_recs = final_recs + fallback_recs[:rec_count - len(final_recs)]

    return ", ".join(final_recs)


# Generate Recommendations
def generate_recommendations(df, features):

    sim_df, top_products = build_similarity_from_uploaded_data(df)

    recommendations = []

    for _, row in features.iterrows():

        cust_id = row["Customer ID"]
        prob = row["Upsell Probability"]

        # Use full Products Purchased from features, not only rec_df sample
        purchased_text = row.get("Products Purchased", "")

        purchased = [
            item.strip()
            for item in str(purchased_text).split(",")
            if item.strip() != ""
        ]

        rec = recommend_for_customer(
            customer_id=cust_id,
            purchased=purchased,
            sim_df=sim_df,
            top_products=top_products,
            upsell_prob=prob
        )

        recommendations.append({
            "Customer ID": cust_id,
            "Recommended Products": rec
        })

    return pd.DataFrame(recommendations)



# PART 3 : Upsell Logic 

# Customer Features

def build_model_features(df):

    import pandas as pd
    import numpy as np


    # -------------------------------
    # CLEANING
    # -------------------------------

    df.columns = df.columns.str.strip().str.upper()

    df["BILL_DATE"] = pd.to_datetime(
        df["BILL_DATE"],
        dayfirst=True,
        errors="coerce"
    )

    df = df.dropna(subset=["BILL_DATE"])

    # Numeric fixes
    numeric_cols = ["NSV", "DISCOUNT", "QTY"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        ).fillna(0)

       
    # -----------------------------------
    # STORE RETURNS / REFUNDS SEPARATELY
    # -----------------------------------

    original_df = df.copy()

    returns_df = original_df[
        original_df["QTY"] <= 0
    ].copy()

    total_returns = returns_df.shape[0]
      
    return_customers = returns_df[
        "CUSTOMER_ID" 
    ].nunique()

    # Remove invalid qty
    df = df[df["QTY"] > 0]

    # -------------------------------
    # GAP DAYS FEATURE
    # -------------------------------

    df = df.sort_values(
        ["CUSTOMER_ID", "BILL_DATE"]
    )

    df["Previous Date"] = df.groupby(
        "CUSTOMER_ID"
    )["BILL_DATE"].shift(1)

    df["Gap Days"] = (
        df["BILL_DATE"] - df["Previous Date"]
    ).dt.days

    # -------------------------------
    # CUSTOMER AGGREGATION
    # -------------------------------

    cust_df = df.groupby("CUSTOMER_ID").agg({

        "NSV": "sum",
        "DISCOUNT": "sum",
        "QTY": "sum",
        "BILL_NO": "nunique",

        "BILL_DATE": ["max", "min"],

        "PRODUCT_DESCRIPTION": lambda x: list(set(x)),

        "GENDER": "first",
        "CITY": "first",
        "REGION": "first",

        "Gap Days": "mean"

    }).reset_index()

    # -------------------------------
    # FLATTEN COLUMNS
    # -------------------------------

    cust_df.columns = [

        "Customer ID",
        "Total Spend",
        "Total Discount",
        "Total Qty",
        "Total Transactions",

        "Last Purchase",
        "First Purchase",

        "Products Purchased",

        "Gender",
        "City",
        "Region",

        "Average Purchase Gap"
    ]

    # -------------------------------
    # RFM FEATURES
    # -------------------------------

    today = df["BILL_DATE"].max()

    cust_df["Recency"] = (
        today - cust_df["Last Purchase"]
    ).dt.days

    cust_df["Tenure"] = (
        cust_df["Last Purchase"]
        - cust_df["First Purchase"]
    ).dt.days

    cust_df["Frequency"] = (
        cust_df["Total Transactions"]
    )

    cust_df["Average Spend"] = np.where(
        cust_df["Total Transactions"] > 0,
        cust_df["Total Spend"] / cust_df["Total Transactions"],
        0
    )

    cust_df["Average Spend"] = (
        cust_df["Average Spend"]
        .fillna(0)
    )

    cust_df["Average Purchase Gap"] = (
        cust_df["Average Purchase Gap"]
        .fillna(0)
    )

    # Unique category count
    unique_cat = df.groupby("CUSTOMER_ID")["PRODUCT_DESCRIPTION"].nunique()

    cust_df["Unique_Categories"] = (
        cust_df["Customer ID"]
        .map(unique_cat)
        .fillna(0)
    )


    # -------------------------------
    # DISCOUNT RATIO
    # -------------------------------

    cust_df["Discount_Ratio"] = np.where(

        cust_df["Total Spend"] > 0,

        cust_df["Total Discount"]
        /
        (cust_df["Total Spend"]+1),

        0
    )

    # -------------------------------
    # CLEAN PRODUCT DISPLAY
    # -------------------------------

    cust_df["Products Purchased"] = (
        cust_df["Products Purchased"]
        .apply(lambda x: ", ".join(map(str, x)))
    )

    # -------------------------------
    # RETURN CUSTOMERS DISPLAY
    # -------------------------------

    cust_df["Return_Products_Count"] = total_returns

    cust_df["Customers_With_Returns"] = return_customers

    return cust_df



# PART 4 : Usage of Python Code

# Generate ML Predictions

import pickle

def generate_ml_predictions(df):

    # -----------------------------------
    # STEP 1 : BUILD FEATURES
    # -----------------------------------

    features = build_model_features(df)

    # -----------------------------------
    # STEP 2 : LOAD MODEL
    # -----------------------------------

    with open("upsellmodel.pkl", "rb") as f:
        model = pickle.load(f)

    with open("upsellmodel_columns.pkl", "rb") as f:
        model_columns = pickle.load(f)

    with open("prediction_threshold.pkl", "rb") as f:
        threshold = pickle.load(f)

    # -----------------------------------
    # STEP 3 : PREPARE MODEL INPUT
    # -----------------------------------

    # Remove non-ML columns
    drop_cols = [

        "Customer ID",
        "Last Purchase",
        "First Purchase",
        "Products Purchased",
        "Gender",
        "City",
        "Region"

    ]

    X = features.drop(
        columns=drop_cols,
        errors="ignore"
    )

    # -----------------------------------
    # STEP 4 : COLUMN ALIGNMENT
    # -----------------------------------

    # Add missing columns
    for col in model_columns:

        if col not in X.columns:
            X[col] = 0

    # Keep exact training order
    X = X[model_columns]

    # Fill missing values
    X = X.fillna(0)

    # -----------------------------------
    # STEP 5 : PREDICT
    # -----------------------------------

    probs = model.predict_proba(X)[:, 1]
    
    preds = (probs >= threshold).astype(int)

    # -----------------------------------
    # STEP 6 : SAVE OUTPUT
    # -----------------------------------

    features["Upsell Probability"] = probs

    features["Upsell Prediction"] = preds

    return features



# PART 6 : Building Output

# Merge Features
def build_final_output(df):

    import pandas as pd
    import numpy as np
    with open("prediction_threshold.pkl", "rb") as f:
        threshold = pickle.load(f)

    # -----------------------------------
    # CLEAN DATA
    # -----------------------------------

    numeric_cols = ["QTY", "NSV", "DISCOUNT"]

    for col in numeric_cols:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        ).fillna(0)

    # -----------------------------------
    # GENERATE ML FEATURES + PREDICTIONS
    # -----------------------------------

    features = generate_ml_predictions(df)

    # -----------------------------------
    # GENERATE RECOMMENDATIONS
    # -----------------------------------

    REC_ROWS = 50000

    rec_df = df.copy()

    rec_df["BILL_DATE"] = pd.to_datetime(
        rec_df["BILL_DATE"],
        dayfirst=True,
        errors="coerce"
    )

    rec_df = (
        rec_df
        .dropna(subset=["BILL_DATE"])
        .sort_values("BILL_DATE", ascending=False)
        .head(REC_ROWS)
    )

    recommendations = generate_recommendations(
        df=rec_df,
        features=features
    )

    # -----------------------------------
    # MERGE RECOMMENDATIONS
    # -----------------------------------

    final = features.merge(
        recommendations,
        on="Customer ID",
        how="left"
    )


    # -----------------------------------
    # CREATE UPSELL CATEGORY
    # -----------------------------------

    final["Upsell Category"] = "Low"

    mask = final["Upsell Probability"] >= threshold


    # # DEBUG CHECK for lesser category segmentation
    # print("\nUpsell Probability Summary")
    # print(
    #     final.loc[mask, "Upsell Probability"].describe()
    # )

    # print("\nUnique Probability Values")
    # print(
    #     final.loc[mask, "Upsell Probability"].nunique()
    # )

    # print("\nTop 20 Probability Values")
    # print(
    #     final.loc[mask, "Upsell Probability"]
    #     .sort_values(ascending=False)
    #     .head(20)
    # )


    try:

        final.loc[mask, "Upsell Category"] = pd.qcut(

            final.loc[mask, "Upsell Probability"],

            q=2,

            labels=["Medium", "High"],

            duplicates="drop"
        )

    except:

        mid = final.loc[
            mask,
            "Upsell Probability"
        ].median()

        final.loc[mask, "Upsell Category"] = pd.cut(

            final.loc[mask, "Upsell Probability"],

            bins=[threshold, mid, float("inf")],

            labels=["Medium", "High"]
        )

    # -----------------------------------
    # DISPLAY %
    # -----------------------------------

    final["Upsell Probability in %"] = (

        (final["Upsell Probability"] * 100)

        .round(2)

        .astype(str)

        + "%"
    )

    # -----------------------------------
    # SORT OUTPUT
    # -----------------------------------

    final = final.sort_values(

        by="Upsell Probability",

        ascending=False
    )

    # -----------------------------------
    # FINAL DISPLAY COLUMNS
    # -----------------------------------

    required_cols = [

        "Customer ID",

        "Upsell Prediction",
        "Upsell Probability",
        "Upsell Probability in %",
        "Upsell Category",

        "Unique_Categories",
        "Products Purchased",
        "Recommended Products",

        "Total Spend",
        "Total Transactions",
        "Average Spend",

        "Recency",
        "Frequency",

        "Discount_Ratio",

        "Return_Products_Count",
        "Customers_With_Returns"

    ]
    
    final = final[

        [col for col in required_cols if col in final.columns]

    ]

    return final



# PART 7 : Run Pipeline

def run_pipeline(
    trans_file=None,
    cust_file=None,
    combined_file=None
):

    import pandas as pd

    # -----------------------------------
    # OPTION 1 : SEPARATE FILES
    # -----------------------------------

    if trans_file is not None and cust_file is not None:

        df_transaction, df_customer = load_data(
            trans_file,
            cust_file
        )

        df_transaction = clean_data(df_transaction)
        df_customer = clean_data(df_customer)

        df = merge_data(
            df_transaction,
            df_customer
        )

    # -----------------------------------
    # OPTION 2 : COMBINED FILE
    # -----------------------------------

    elif combined_file is not None:

        df = pd.read_csv(combined_file)

        df = clean_data(df)

    # -----------------------------------
    # NO FILE PROVIDED
    # -----------------------------------

    else:

        raise ValueError(
            "No valid input files provided"
        )
    

    # -----------------------------------
    # Visible in Terminal
    # -----------------------------------
    print("Step 1: Data Loaded")

    features = generate_ml_predictions(df)
    print("Step 2: Predictions Complete")

    recommendations = generate_recommendations(
        df=df,
        features=features
    )
    print("Step 3: Recommendations Complete")

    # -----------------------------------
    # BUILD FINAL OUTPUT
    # -----------------------------------

    final = features.merge(
        recommendations,
        on="Customer ID",
        how="left"
    )
    print("Step 4: Merge Complete")

    final_df = build_final_output(df)

    return final_df