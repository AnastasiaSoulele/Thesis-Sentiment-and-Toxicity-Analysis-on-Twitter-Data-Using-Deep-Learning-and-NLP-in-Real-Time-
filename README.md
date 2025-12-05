# Twitter Sentiment & Toxicity Analysis (SentimentX)

This project implements **real-time sentiment, emotion, and toxicity analysis** on Twitter data 
using deep learning models (DistilBERT, DistilRoBERTa, Toxic-BERT) and NLP techniques.  
It includes both an **interactive web dashboard** and an **offline analyzer** for batch processing.

---

## Requirements

- Python 3.9+  
- pip (Python package manager)  
- (Optional) GPU with CUDA for faster model inference  

---

## Setup

It is recommended to create a clean virtual environment.

### 1. Create environment
```bash
python -m venv venv
```

### 2. Activate environment
- On Linux/Mac:
  ```bash
  source venv/bin/activate
  ```
- On Windows:
  ```bash
  venv\Scripts\activate
  ```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Application

### Option A: Interactive Web Dashboard
1. Activate the virtual environment.  
2. Run the Flask app:
   ```bash
   python app.py
   ```
3. Open your browser at:
   ```
   http://127.0.0.1:5000
   ```
   or replace `127.0.0.1` with your server’s IP if accessing remotely.
