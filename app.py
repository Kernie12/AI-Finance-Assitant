import gradio as gr
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import altair as alt
import cohere

df = pd.read_csv("/Users/macbook/Desktop/Kenny_cleaned_bankstatement.csv")

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])
df['amount'] = df['amount'].astype(float)
df['year_month'] = df['Date'].dt.to_period('M')

def detect_fraud():
    temp_df = df.copy()
    temp_df['dayofweek'] = temp_df['Date'].dt.dayofweek
    temp_df['day'] = temp_df['Date'].dt.day
    temp_df['month'] = temp_df['Date'].dt.month
    X = temp_df[['amount', 'dayofweek', 'day', 'month']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    temp_df['cluster'] = kmeans.fit_predict(X_scaled)
    centroids = kmeans.cluster_centers_
    temp_df['distance_to_center'] = np.linalg.norm(X_scaled - centroids[temp_df['cluster']], axis=1)
    threshold = temp_df['distance_to_center'].quantile(0.95)
    outliers = temp_df[temp_df['distance_to_center'] > threshold]
    return outliers[['Date', 'description_clean', 'amount', 'distance_to_center']]

# --- Budget Planner ---
def budget_summary():
    summary = df.groupby('year_month')['amount'].sum().reset_index()
    summary['year_month'] = summary['year_month'].astype(str)
    chart = alt.Chart(summary).mark_line(point=True).encode(
        x=alt.X('year_month:T', title='Month'),
        y=alt.Y('amount:Q', title='Total Spending'),
        tooltip=['year_month', 'amount']
    ).properties(title="📅 Monthly Spending Summary").configure_axis(
        labelFontSize=12,
        titleFontSize=14
    )
    return chart

# --- Top Categories ---
def top_categories():
    top = df['category'].value_counts().nlargest(10).reset_index()
    top.columns = ['Category', 'Count']
    chart = alt.Chart(top).mark_bar().encode(
        x=alt.X('Count:Q', title='Transaction Count'),
        y=alt.Y('Category:N', sort='-x', title='Category')
    ).properties(title="📊 Top 10 Transaction Categories").configure_axis(
        labelFontSize=12,
        titleFontSize=14
    )
    return chart

co = cohere.Client("U7gUj0Z8JRfky9hJnFJ2zOvhsaA9KZgZRDIS6hBN")  

def ask_advisor(question):
    try:
        # Format the entire dataset (compactly)
        context = "\n".join([
            f"{row['Date']} | {row['description_clean']} | ₦{row['amount']:.2f} | {row['category']}"
            for _, row in df.iterrows()
        ])

        prompt = f"""
You're a professional financial advisor. Based on the following bank transactions, provide smart, actionable, and concise advice. Do not repeat the transactions. Just answer the user’s question intelligently.

Transaction Summary:
{context}

User Question: {question}
"""
        response = co.chat(message=prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"
def dashboard():
    return top_categories()

def planner():
    return budget_summary()

def full_app(question):
    fraud = detect_fraud()
    response = ask_advisor(question)
    return fraud.head(10), response

with gr.Blocks(title="AI Personal Finance Dashboard", css="""
    .gradio-container {padding: 10px;}
    .gr-tabitem {padding: 10px;}
    h1, h3, p {text-align: center; font-family: 'Segoe UI', sans-serif;}
    .gr-button {width: 100%; font-size: 16px;}
    @media (max-width: 768px) {
        h1 {font-size: 22px;}
        h3 {font-size: 18px;}
        .gr-button {font-size: 14px;}
    }
""") as app:
    gr.Markdown("""
    <h1>💸 AI Personal Finance Dashboard</h1>
    <p>Analyze your spending, detect anomalies, and ask an AI for financial advice!</p>
    """)

    with gr.Tabs():
        with gr.Tab("📊 Dashboard"):
            gr.Markdown("<h3>Top Spending Categories</h3>")
            gr.Plot(label="Category Chart", value=dashboard)

        with gr.Tab("🔍 Fraud Detection"):
            gr.Markdown("<h3>Suspicious Transactions</h3>")
            gr.DataFrame(value=detect_fraud, label="Potential Outliers")

        with gr.Tab("📅 Budget Planner"):
            gr.Markdown("<h3>Monthly Spending Trends</h3>")
            gr.Plot(label="Monthly Line Chart", value=planner)

        with gr.Tab("🤖 AI Financial Advisor"):
            gr.Markdown("<h3>Ask the AI about your Finances</h3>")
            question = gr.Textbox(label="Type your question here")
            advisor_out = gr.Textbox(label="AI Response")
            submit_btn = gr.Button("💬 Get Advice")
            submit_btn.click(fn=full_app, inputs=question, outputs=[gr.DataFrame(), advisor_out])

app.launch(share=True)
