
# To activate our APP through terminal : python -m streamlit run app.py
# To use this install Streamlit : pip install streamlit


# Libraries
import streamlit as st
from model_pipeline import run_pipeline



# PART 1 : Title of the Page

st.set_page_config(page_title="Machine Lead Upsell Model with Recommendations", layout="wide")

st.title("🛍️ Machine Lead Upsell Model with Recommendations")



# PART 2 : Sidebar

st.sidebar.title("📊 Upsell Dashboard Guide")

# 1. How to Use

with st.sidebar.expander("📖 How to Use", expanded=True):
    st.write("""
    1. Upload transaction data and customer data  or combined file
    2. Click **Run Model**  
    3. View customer-wise upsell insights  
    4. Check recommendations and download results 
    """)

# 2. How to Use Insights 

with st.sidebar.expander("🎯 Upsell Target Insights"):
    st.write("""
    🔥 **High** → Prioritize for premium products and bundles  
    ⚖️ **Medium** → Target with offers, combos, and cross-sell campaigns  
    ❄️ **Low** → Engage with awareness, discounts, or retention campaigns  

    📌 Goal: Identify customers with higher upsell potential and recommend suitable products.
    """)

# 3. Features

with st.sidebar.expander("⚙️ Features"):
    st.write("""
    ✔ Customer-level upsell prediction  
    ✔ Upsell probability scoring  
    ✔ Personalized product recommendations  
    ✔ Unique category and purchase behavior analysis  
    ✔ Interactive charts and downloadable results   
    """)

# 4. Model Info

with st.sidebar.expander("🧠 Model Info"):
    st.write("""
    - Model: Random Forest Classifier  
    - Evaluation: ROC-AUC, classification report, and confusion matrix  
    - Prediction Threshold: 60%  
    - Recommendation Engine: Item-based similarity system  
    - System Type: ML + Recommendation Hybrid Pipeline  

    📌 Based on:
    - Recency  
    - Average purchase gap  
    - Unique product categories  
    - Discount ratio  
    - Growth Rate
    """)

# 5. Data Summary (Dynamic)

with st.sidebar.expander("📊 Data Summary", expanded=True):

    if "final_df" in st.session_state:

        df = st.session_state["final_df"]

        if df is not None:

            df.columns = df.columns.str.strip()

            st.success("👀 Data Loaded")

            st.write(f"👥 Customers: {df.shape[0]}")

            if "Total Spend" in df.columns:
                    st.write(f"Average Spend: ₹ {int(df['Total Spend'].mean())}")

            if "Upsell Prediction" in df.columns:
                    high = df["Upsell Prediction"].sum()
                    st.write(f"Customers with Upsell 1 : {int(high)}")
                    

        else:
            st.warning("⚠️ Data is empty")

    else:
        st.info("📂 Upload files and run model to see summary")

# 6. Feedback

import datetime
import os

with st.sidebar.expander("💬 Feedback"):

    feedback = st.text_area("Share your thoughts:")

    if not os.path.exists("feedback_log.csv"):
        with open("feedback_log.csv", "w") as f:
            f.write("timestamp,feedback\n")

    if st.button("Submit Feedback", key="feedback_btn"):
        if feedback.strip():

            with open("feedback_log.csv", "a") as f:
                f.write(f"{datetime.datetime.now()},{feedback.strip()}\n")

            st.success("Thanks for your feedback! 💙")
            st.toast("Feedback submitted!") 

        else:
            st.warning("Please enter feedback before submitting")

    st.caption("Your feedback helps improve the model")

# 7. Footer

st.sidebar.markdown("---")
st.sidebar.caption("Version 1.0 | Built with Streamlit")
st.sidebar.caption("Developed by Shashwati Saha ✨⭐")



# PART 3 : Uploads & Run

st.write("Upload your raw data and get customer-wise upsell & product insights")


# Mode selection
st.markdown("Choose Data Input Method suitable for your data :")
st.info("Large datasets may take longer to process. Processing may take 2–3 minutes due to recommendation generation.")

mode = st.radio(
    "",
    ["📂 Upload Separate Files", "📊 Upload Combined File"],
 horizontal=True
)

st.caption("💡 Use combined file if your dataset already contains customer + transaction info")

# SEPARATE FILES
if mode == "📂 Upload Separate Files":

    st.write("### Upload Separate Files")
    
    col1, col2 = st.columns(2)

    with col1:
        trans_file = st.file_uploader(
            "📂 Upload Transactions CSV",
            type=["csv"],
            key="trans_file"
        )
    with col2:
        cust_file = st.file_uploader(
            "📂 Upload Customers CSV",
            type=["csv"],
            key="cust_file"
        )

    if trans_file is not None and cust_file is not None:
        st.success("✅ Files uploaded successfully!")

        if st.button("🚀 Run Model", key="run_sep"):
                
                try:

                    with st.spinner("Processing data..."):
                        final_df = run_pipeline(
                            trans_file=trans_file,
                            cust_file=cust_file
                        )
                        st.session_state["final_df"] = final_df
                        st.success("✅ Model executed successfully!")
                        st.rerun()

                except Exception as e:

                    st.error(f"❌ Error:{e}")

# COMBINED FILE
elif mode == "📊 Upload Combined File":

    st.write("### Upload Combined File")

    combined_file = st.file_uploader(
        "📊 Upload Combined File",
        type=["csv"],
        key="combined_file"
    )

    if combined_file is not None:
        st.success("✅ File uploaded successfully!")

        if st.button("🚀 Run Model", key="run_combined"):
            
            try:

                with st.spinner("Processing data..."):
                    final_df = run_pipeline(
                        combined_file=combined_file
                    )
                    st.session_state["final_df"] = final_df
                    st.success("✅ Model executed successfully!")
                    st.rerun()

            except Exception as e:

                if "bins must increase monotonically" in str(e):
                    st.error(
                        "⚠️ Category segmentation could not be created because the upsell probabilities are too similar in this dataset. "
                        "Please try another dataset with more variation, or review the probability distribution."
                    )
            else:
                    st.error(f"❌ Error: {e}")

       


# PART 4 : Processes after SUCCESSFUL uploads [Very IMP Part]

# EVERYTHING BELOW RUNS FROM STORED DATA

final_df = st.session_state.get("final_df")


if final_df is not None and not final_df.empty:


    # 1. KPIs

    st.subheader("📊 Summary Insights")

    total_customers = final_df.shape[0]

    total_transactions = (
        int(final_df["Total Transactions"].sum())
        if "Total Transactions" in final_df.columns else 0
    )

    high_count = (
        (final_df["Upsell Category"] == "High").sum()
        if "Upsell Category" in final_df.columns else 0
    )

    total_returns = (
        int(final_df["Return_Products_Count"].fillna(0).iloc[0])
        if "Return_Products_Count" in final_df.columns else 0
    )

    return_customers = (
        int(final_df["Customers_With_Returns"].fillna(0).iloc[0])
        if "Customers_With_Returns" in final_df.columns else 0
    )

    total = total_customers if total_customers > 0 else 1

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("👥 Total Customers", total_customers)

    with col2:
        st.metric("🧾 Total Transactions", total_transactions)

    with col3:
        st.metric(
            "🔥 High Potential",
            f"{high_count}",
            f"{(high_count / total) * 100:.1f}% of customers"
        )

    with col4:
        st.metric("↩️ Returned Products Count", total_returns)

    with col5:
        st.metric("👥 Return Customers", return_customers)

    # 2. Pie Chart
    import plotly.express as px

    st.subheader("📊 Upsell Category Distribution")

    # Prepare data

    if "Upsell Category" in final_df.columns and not final_df.empty:

        category_counts = (
            final_df["Upsell Category"]
            .value_counts()
            .reset_index()
        )

        category_counts.columns = ["Category", "Count"]


        # Create pie chart
        fig = px.pie(
            category_counts,
            names="Category",
            values="Count",
            hole=0.5
        )

        fig.update_traces(
            textinfo='percent+label',
            textfont_size=14
        )

        fig.update_layout(
            showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(orientation="h", y=-0.1)
        )

        # Layout
        col1, col2 = st.columns([2, 1])

        with col1:
            st.plotly_chart(fig, use_container_width=True)   

        # Insights (description)
        with col2:
            st.markdown("### 📌 Insights")

            high = (final_df["Upsell Category"] == "High").sum()
            medium = (final_df["Upsell Category"] == "Medium").sum()
            low = (final_df["Upsell Category"] == "Low").sum()
            total = final_df.shape[0]

            st.write(f"""
            🔥 **High Potential Customers:** {high}  
            ⚖️ **Medium Potential:** {medium}  
            ❄️ **Low Potential:** {low}  

            👉 {(high_count / total) * 100:.1f}% of customers are strong upsell targets.

            💡 Focus marketing efforts on **High category** for better ROI.
            """)

            st.info(f"""
            💡 Recommendation:
            Prioritize high upsell customers for targeted campaigns.
            """)

    else:
        st.warning("Upsell Category not available or data is empty.")


    # 3. Bar Graph
  
    st.subheader("📊 Upsell Distribution (0 vs 1)")

    upsell_counts = final_df["Upsell Prediction"].value_counts()


    upsell_df = upsell_counts.reset_index()
    upsell_df.columns = ["Upsell", "Count"]
    upsell_df = upsell_df.sort_values(by="Count", ascending=True)

    total = upsell_df["Count"].sum()

    upsell_df["Percentage"] = (
        (upsell_df["Count"] / total) * 100
    ).round(1).astype(str) + "%"

    # Map labels
    upsell_df["Upsell"] = upsell_df["Upsell"].map({
        1: "Upsell 1",
        0: "Upsell 0"
    })

    import plotly.express as px

    fig = px.bar(
        upsell_df,
        x="Count",
        y="Upsell",
        orientation='h',
        text="Percentage"
    )

    fig.update_traces(
        textposition='outside'
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Number of Customers",
        yaxis_title="",
    )

    # Layout (like pie chart)
    col1, col2 = st.columns([2, 1])

    with col1:
        st.plotly_chart(fig, use_container_width=True)

    # Insights (description)
    with col2:
        st.markdown("### 📌 Insights")

        high = (final_df["Upsell Prediction"] == 1).sum()
        low = (final_df["Upsell Prediction"] == 0).sum()
        total = final_df.shape[0]

        st.write(f"""
        🔥 **Upsell 1 :** {high}  
        ❄️ **Upsell 0 :** {low}  

        👉 {round((high/total)*100)}% customers are likely to upsell.

        💡 Focus on converting **Low → High Upsell** through targeted strategies.
        """)

        st.info(f"""
        💡 Strategy:
        Use recommendations to convert low-potential customers to higher potential.
        """)


    # 4. Top Customers
    st.subheader("🏆 Top Upsell Customers")

    # User input (default = 10)
    total_customers = final_df.shape[0]

    top_n = st.number_input(
        "Select number of top customers to display",
        min_value=1,
        max_value=total_customers,
        value=min(10, total_customers),  # smart default
        step=1
    )

    # Sort using numeric column (IMPORTANT)
    top_customers = final_df.sort_values(
        by="Upsell Probability",
        ascending=False
    ).head(top_n)[[
        "Customer ID",
        "Upsell Probability in %",
        "Products Purchased",
        "Recommended Products"
    ]]

    # Add Serial Number(optional)
    top_customers.insert(0, "S.No", range(1, len(top_customers) + 1))

    st.dataframe(top_customers, hide_index=True)


    # 5. Search
    st.subheader("🔍 Search Customer")

    search_id = st.text_input("Enter Customer ID")

    if search_id:
        
        filtered = final_df[
            final_df["Customer ID"]
            .astype(str)
            .str.strip()
            .str.upper()
            ==
            search_id.strip().upper()
        ]

        if not filtered.empty:
                st.dataframe(filtered)
        else:
                st.warning("No customer found")

    st.subheader("🎯 AI-Generated Upsell Recommendations")
    

    # 6. Results
    st.subheader("📊 Final Customer Insights")
    
    display_df = final_df.drop(columns=["Upsell Probability","Discount_Ratio",
    "Return_Products_Count","Customers_With_Returns","Unique_Categories"], errors="ignore")

    st.dataframe(display_df, use_container_width=True, hide_index=True)


    # 7. Download Results
    csv = final_df.to_csv(index=False).encode('utf-8')

    st.download_button(
            "📥 Download Upsell Customer Insights",
            csv,
            "upsell_output.csv",
            "text/csv"
    )



# RUNS WHEN NO DATA UPLOADED
else:
    st.info("👆 Upload your data and run the model to unlock insights!")