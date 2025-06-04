# 💰 AI-Powered Personal Finance Assistant

A smart, user-friendly financial dashboard that processes bank statements—even in image-based PDF format—and delivers actionable insights through AI-powered analysis, fraud detection, and budgeting. Built with data science, NLP, and OCR, this app makes personal finance management smarter and stress-free.

---

## 🚀 Features

### 🔍 OCR-Powered Data Extraction
- Converts **image-based PDF bank statements** into structured data.
- Uses `pdf2image` + `Tesseract OCR` for text recognition.

### 🧹 Data Cleaning & Categorization
- Extracts and standardizes **date**, **description**, and **amount**.
- Categorizes transactions into **spending groups** (e.g. rent, salary, groceries).

### 🧠 Unsupervised Fraud Detection
- Detects **anomalous or suspicious transactions** using time-aware clustering and Isolation Forest.

### 💬 AI-Powered Financial Advisor
- Powered by Cohere’s LLM (Command-R+).
- Ask natural language questions like:  
  > *"How much did I spend on food last month?"*

### 📊 Budget Planning
- Recommends a **monthly budget** based on historical spending.
- Identifies **saving opportunities**.

### 📈 Account Balance Forecasting
- Predicts your future account balance using regression or time-series modeling.

---

## 🛠 Tech Stack

- Python
- Streamlit
- Pandas, NumPy, Scikit-learn
- Tesseract OCR + pdf2image
- Cohere (LLM API)
- Matplotlib / Plotly (for visuals)

---

## 🧪 How It Works

### 1. OCR + PDF Processing
```python
from pdf2image import convert_from_path
import pytesseract
