import os
import json
import re
from db import get_db_connection
from transformers import pipeline, DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import numpy as np

# === 1. Device ===
device_idx = 0 if torch.cuda.is_available() else -1
torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 2. Φόρτωση ΜΟΝΤΕΛΩΝ ===
# 2a)  DistilBERT για SENTIMENT
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fine_tuned_bert")

sent_tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
sent_model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR).to(torch_device)
sent_model.eval()

def sentiment_analyzer(text):
    """
    Επιστρέφει ίδιο schema με HF pipeline: [{'label': 'Positive'|'Negative', 'score': float}]
    """
    # batchify αν περαστει ΛΙΣΤΑ
    if isinstance(text, list):
        outputs = []
        # μικρό batching για GPU αποδοτικότητα
        batch_size = 32
        for i in range(0, len(text), batch_size):
            batch_texts = text[i:i+batch_size]
            enc = sent_tokenizer(
                batch_texts, truncation=True, padding=True, max_length=128, return_tensors="pt"
            ).to(torch_device)
            with torch.no_grad():
                logits = sent_model(**enc).logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()  # shape: [B, 2]
            for p in probs:
                pred_id = int(np.argmax(p))
                label = "Positive" if pred_id == 1 else "Negative"
                score = float(p[pred_id])
                outputs.append({"label": label, "score": score})
        return outputs
    else:
        enc = sent_tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt").to(torch_device)
        with torch.no_grad():
            logits = sent_model(**enc).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # [2]
        pred_id = int(np.argmax(probs))
        label = "Positive" if pred_id == 1 else "Negative"
        score = float(probs[pred_id])
        return [{"label": label, "score": score}]

# 2b) EMOTION (μένει pipeline)
emotion_analyzer = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    device=device_idx
)

# 2c) TOXICITY (μένει pipeline)
toxicity_analyzer = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    device=device_idx
)

# === 3. Βοηθητικά ===
def extract_hashtags(text):
    return ' '.join(re.findall(r"#\w+", text))

# === 4. Σύνδεση με ΒΔ ===
conn = get_db_connection()
cursor = conn.cursor()

# === 5. Διαδρομή batches ===
batch_folder = "fake_batches"
files = sorted(os.listdir(batch_folder))

inserted = 0

for filename in files:
    if not filename.endswith(".json"):
        continue

    with open(os.path.join(batch_folder, filename), "r", encoding="utf-8") as f:
        tweets = json.load(f)

    for i, tweet in enumerate(tweets, 1):
        print(f"Analyzing tweet {i}/{len(tweets)} from {filename}...")
        tweet_id = tweet["id"]
        username = tweet["username"]
        text = tweet["text"]
        hashtags = extract_hashtags(text)

        try:
            # === Sentiment: my model ===
            sentiment_result = sentiment_analyzer(text)[0]  # {'label','score'}

            # === Emotion / Toxicity ===
            emotion_result = emotion_analyzer(text)[0]      # {'label','score'}
            toxicity_result = toxicity_analyzer(text)[0]    # [{'label',...}] -> top 1

            cursor.execute("""
                INSERT INTO tweet_analysis (
                    tweet_id, username, tweet_text,
                    sentiment_label, sentiment_score,
                    emotion_label,
                    toxicity_level, toxicity_score,
                    hashtags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                tweet_id, username, text,
                sentiment_result["label"], float(sentiment_result["score"]),
                emotion_result["label"],
                toxicity_result["label"], float(toxicity_result["score"]),
                hashtags
            ))

            inserted += 1

        except Exception as e:
            print(f"Σφάλμα στο tweet {tweet_id}: {e}")

# === Κλείσιμο ===
conn.commit()
cursor.close()
conn.close()

print(f"Ολοκληρώθηκε η ανάλυση. Tweets που αποθηκεύτηκαν: {inserted}")
