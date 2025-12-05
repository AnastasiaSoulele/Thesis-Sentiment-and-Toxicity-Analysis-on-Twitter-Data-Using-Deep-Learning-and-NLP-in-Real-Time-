from flask import Flask, render_template, request, jsonify
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer as HFTokenizer, AutoModelForSequenceClassification as HFModel
import torch
import shap
import os
import re
import json
import html
import emoji
from db import get_db_connection
from collections import Counter, defaultdict
import numpy as np

app = Flask(__name__)

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load fine-tuned sentiment model
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "fine_tuned_bert")

tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()

# -------------------------
# SHAP (word-level, όχι wordpieces)
# -------------------------
EXPLAIN_TOKEN_PATTERN = r"[A-Za-z]+(?:'[A-Za-z]+)?|[#@]\w+|\d+(?:\.\d+)?|[!?.,:;()]"

def _simple_word_tokenizer(s: str):
    return re.findall(EXPLAIN_TOKEN_PATTERN, s)

classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True,
    device=0 if torch.cuda.is_available() else -1
)
explainer = shap.Explainer(
    classifier,
    shap.maskers.Text(EXPLAIN_TOKEN_PATTERN),
    algorithm="partition"
)

# -------------------------
# Emotion model
# -------------------------
emotion_model_name = "j-hartmann/emotion-english-distilroberta-base"
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name).to(device)
emotion_classifier = pipeline(
    "text-classification",
    model=emotion_model,
    tokenizer=emotion_tokenizer,
    return_all_scores=True,
    device=0 if torch.cuda.is_available() else -1
)

# -------------------------
# Toxicity model 
# -------------------------
toxicity_model_name = "unitary/toxic-bert"
toxicity_tokenizer = HFTokenizer.from_pretrained(toxicity_model_name)
toxicity_model = HFModel.from_pretrained(toxicity_model_name).to(device)
toxicity_classifier = pipeline(
    "text-classification",
    model=toxicity_model,
    tokenizer=toxicity_tokenizer,
    return_all_scores=True,
    device=0 if torch.cuda.is_available() else -1
)

# -------------------------
# Dynamic stopwords for SHAP aggregation
# -------------------------
BASE_STOPWORDS = {
    "the","is","a","an","and","or","in","on","at","of","to","for","this","that",
    "with","you","i","it","what","how","was","were","be","are","they","we","he","she"
}
_dynamic_stopwords_cache = None

def build_dynamic_stopwords(limit=100):
    """Μαζεύει συχνά tokens από τη DB ώστε να τα αγνοούμε στις εξηγήσεις."""
    global _dynamic_stopwords_cache
    if _dynamic_stopwords_cache is not None:
        return _dynamic_stopwords_cache

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT tweet_text FROM tweet_analysis")
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    freq = Counter()
    for (txt,) in rows:
        if not txt:
            continue
        toks = re.findall(EXPLAIN_TOKEN_PATTERN, str(txt).lower())
        for t in toks:
            if t.startswith("#") or t.startswith("@"):
                continue
            if len(t) <= 1 or re.fullmatch(r"[!?.,:;()]", t):
                continue
            freq[t] += 1

    topN = {w for w, _ in freq.most_common(limit)}
    _dynamic_stopwords_cache = BASE_STOPWORDS | topN
    return _dynamic_stopwords_cache

# -------------------------
# Preprocessing 
# -------------------------
_hashtag_camel_re = re.compile(r"(?<=[a-z])(?=[A-Z])")
URL_RE   = re.compile(r"http\S+|www\S+|https\S+", flags=re.IGNORECASE)
MENT_RE  = re.compile(r"@\w+")
HASH_RE  = re.compile(r"#\w+")
SPACE_RE = re.compile(r"\s+")

def _split_hashtag(tag: str) -> str:
    core = tag[1:] if tag.startswith("#") else tag
    core = _hashtag_camel_re.sub(" ", core)
    core = core.replace("_", " ")
    return core

def _squeeze_repeats(w: str) -> str:
    # κρατάμε μέχρι 2 συνεχόμενους ίδιους χαρακτήρες
    return re.sub(r"(.)\1{2,}", r"\1\1", w)

def clean_for_sentiment(text: str) -> str:
    """
    Cleaning συμβατό με το training:
    - HTML unescape
    - emoji.demojize -> λεκτικές περιγραφές
    - URLs -> space
    - @mentions -> @user
    - #hashtags -> σπάσιμο σε λέξεις
    - κρατάμε ! ?
    - συμπίεση επαναλήψεων
    - lowercase
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = html.unescape(text)
    # demojize (χαμηλό κόστος για μικρά inputs, κρίσιμο για συνέπεια)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = URL_RE.sub(" ", text)
    text = MENT_RE.sub(" @user ", text)
    text = HASH_RE.sub(lambda m: " " + _split_hashtag(m.group(0)) + " ", text)
    # επιτρέπουμε γράμματα/αριθμούς/κενά/!/? , αφαιρούμε λοιπά
    text = re.sub(r"[^A-Za-z0-9\s!?]", " ", text)
    toks = [_squeeze_repeats(t) for t in text.split()]
    text = SPACE_RE.sub(" ", " ".join(toks)).strip().lower()
    return text

def clean_light(text: str) -> str:
    """
    Ελαφρύ cleaning για χαμηλό latency (emotion-toxicity):
    - URLs -> space
    - trim πολλαπλών κενών
    (κρατάμε mentions/emojis/hashtags όπως είναι για να μη χαθεί σήμα)
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = URL_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()

def clean_text(text: str) -> str:
    return clean_for_sentiment(text)

# -------------------------
# Predictors
# -------------------------
def predict_sentiment(text: str) -> str:
    inputs = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    return "Positive" if prediction == 1 else "Negative"

def explain_prediction(cleaned_text: str, top_k=3) -> str:
    """
    ΔΕΧΕΤΑΙ ΗΔΗ ΚΑΘΑΡΙΣΜΕΝΟ ΚΕΙΜΕΝΟ (clean_for_sentiment),
    και εξηγεί σε επίπεδο λέξης γιατί το μοντέλο έβγαλε Positive οrNegative.
    """
    pred_label = predict_sentiment(cleaned_text)
    pred_class = 1 if pred_label == "Positive" else 0

    sv = explainer([cleaned_text])
    tokens = sv.data[0]
    values = sv.values[0]

    if isinstance(values[0], (list, np.ndarray)):
        values = np.array(values)[:, pred_class]

    stopset = build_dynamic_stopwords(limit=100)

    agg = defaultdict(float)
    for tok, val in zip(tokens, values):
        t = str(tok).strip().lower()
        if not t or t in stopset or len(t) <= 1:
            continue
        agg[t] += float(val)

    def _display_token(w: str) -> str:
        return _squeeze_repeats(w)

    sorted_items = sorted(agg.items(), key=lambda x: abs(x[1]), reverse=True)
    supportive = [_display_token(w) for (w, v) in sorted_items if v > 0][:top_k]
    contradictory = [_display_token(w) for (w, v) in sorted_items if v < 0][:top_k]

    explanation = f"The model predicted {pred_label} sentiment mainly because of words: "
    explanation += (", ".join(f"'{w}'" for w in supportive) + ". ") if supportive \
                   else "no clear strong supporting words. "
    if contradictory:
        explanation += "Some other words slightly pushed sentiment towards the opposite side: "
        explanation += ", ".join(f"'{w}'" for w in contradictory) + "."
    return explanation

def predict_emotion(text: str):
    results = emotion_classifier(text)[0]
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    top_emotion = sorted_results[0]["label"]
    return {"emotion": top_emotion, "scores": sorted_results}

def predict_toxicity(text: str) -> float:
    result = toxicity_classifier(text)[0]
    toxic_score = next((x['score'] for x in result if x['label'].lower() == 'toxic'), 0.0)
    return toxic_score

# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    raw_tweet = data.get("tweet", "")

    # 1) Sentiment/SHAP -> heavy-clean (συνεπές με training)
    cleaned_sent = clean_for_sentiment(raw_tweet)
    sentiment = predict_sentiment(cleaned_sent)
    reason = explain_prediction(cleaned_sent)

    # 2) Emotion/Toxicity -> light-clean για χαμηλό latency 
    aux_input = clean_light(raw_tweet)
    emotion_result = predict_emotion(aux_input)
    toxicity = predict_toxicity(aux_input)

    warnings = []
    if len(cleaned_sent.split()) < 3:
        warnings.append("The input text is very short. The sentiment prediction may be inaccurate without more context.")
    if toxicity > 0.5:
        warnings.append("This tweet contains offensive/toxic language.")

    response = {
        "sentiment": sentiment,
        "original": raw_tweet,
        "cleaned": cleaned_sent,                 #sentiment-cleaned
        "reason": reason,
        "emotion": emotion_result["emotion"],
        "emotion_scores": emotion_result["scores"],
        "toxicity": round(toxicity, 3),
        "warnings": warnings if warnings else []
    }
    return jsonify(response)

# Live feed από Twitter (τυχαίο batch όταν δεν δίνεται ?batch=)
@app.route("/live-feed")
def live_feed():
    batch_param = request.args.get("batch", type=int)
    batch_dir = os.path.join(BASE_DIR, "fake_batches")
    if not os.path.isdir(batch_dir):
        return jsonify({"tweets": [], "hasMore": False, "served_batch": None})

    files = [f for f in os.listdir(batch_dir) if f.startswith("batch_") and f.endswith(".json")]
    if not files:
        return jsonify({"tweets": [], "hasMore": False, "served_batch": None})

    import random
    if batch_param:
        batch_name = f"batch_{batch_param}.json"
        chosen = batch_name if batch_name in files else random.choice(files)
    else:
        chosen = random.choice(files)

    batch_file = os.path.join(batch_dir, chosen)
    with open(batch_file, "r", encoding="utf-8") as f:
        tweets = json.load(f)

    has_more = len(files) > 1

    try:
        served_batch = int(re.findall(r"batch_(\d+)\.json", chosen)[0])
    except Exception:
        served_batch = None

    return jsonify({"tweets": tweets, "hasMore": has_more, "served_batch": served_batch})

@app.route("/stats")
def stats():
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1) Πλήθος συνολικών tweets
    cursor.execute("SELECT COUNT(*) FROM tweet_analysis")
    total = cursor.fetchone()[0]

    # 2) Πλήθος ανά sentiment
    cursor.execute("SELECT sentiment_label, COUNT(*) FROM tweet_analysis GROUP BY sentiment_label")
    sentiment_counts = {row[0]: row[1] for row in cursor.fetchall()}

    # 3) Πλήθος ανά emotion
    cursor.execute("SELECT emotion_label, COUNT(*) FROM tweet_analysis GROUP BY emotion_label")
    emotion_counts = {row[0]: row[1] for row in cursor.fetchall()}

    # 4) Πλήθος ανά toxicity label
    cursor.execute("SELECT toxicity_level, COUNT(*) FROM tweet_analysis GROUP BY toxicity_level")
    toxicity_counts = {row[0]: row[1] for row in cursor.fetchall()}

    # 5) Top hashtags (global)
    cursor.execute("SELECT hashtags FROM tweet_analysis")
    hashtag_data = cursor.fetchall()

    global_hashtag_counter = Counter()
    for row in hashtag_data:
        if row[0]:
            for tag in row[0].split():
                tag = tag.strip().lower()
                if tag.startswith('#') and len(tag) > 1:
                    global_hashtag_counter[tag] += 1

    top_hashtags = global_hashtag_counter.most_common(10)
    top_hashtag_list = [tag for tag, _ in top_hashtags]

    # 6) Emotion - Hashtags (μόνο για top hashtags)
    cursor.execute("SELECT emotion_label, hashtags FROM tweet_analysis")
    emotion_hashtag_data = cursor.fetchall()

    emotion_hashtag_map = defaultdict(Counter)
    for emotion, hashtags in emotion_hashtag_data:
        if hashtags:
            for tag in hashtags.split():
                tag = tag.strip().lower()
                if tag in top_hashtag_list:
                    emotion_hashtag_map[emotion][tag] += 1

    emotion_top_hashtags = {
        emotion: [(tag, count) for tag, count in counter.items() if tag in top_hashtag_list]
        for emotion, counter in emotion_hashtag_map.items()
    }

    # 7) Μέση τιμή toxicity_score ανά emotion
    cursor.execute("""
        SELECT emotion_label, AVG(toxicity_score)
        FROM tweet_analysis
        GROUP BY emotion_label
    """)
    avg_toxicity_by_emotion = {row[0]: round(row[1], 4) for row in cursor.fetchall()}

    # 8) Hashtag → sentiment ratio
    cursor.execute("SELECT sentiment_label, hashtags FROM tweet_analysis")
    rows = cursor.fetchall()

    per_tag_counts = defaultdict(lambda: {"Positive": 0, "Negative": 0})

    def _norm_sentiment(s):
        if not s:
            return None
        s = s.strip().lower()
        if "pos" in s:
            return "Positive"
        if "neg" in s:
            return "Negative"
        return None

    top_set = set(top_hashtag_list)
    for s, tags in rows:
        s_norm = _norm_sentiment(s)
        if not tags or s_norm not in ("Positive", "Negative"):
            continue
        for t in str(tags).split():
            tt = t.strip().lower()
            if tt in top_set:
                per_tag_counts[tt][s_norm] += 1

    hashtag_sentiment_ratio = []
    for tag in top_hashtag_list:
        pos = per_tag_counts[tag]["Positive"]
        neg = per_tag_counts[tag]["Negative"]
        tot = pos + neg
        ratio = (pos / tot) if tot > 0 else 0.0
        hashtag_sentiment_ratio.append({
            "hashtag": tag,
            "pos": pos,
            "neg": neg,
            "pos_ratio": round(ratio, 4)
        })

    # 9) Toxicity histogram (bins)
    cursor.execute("SELECT toxicity_score FROM tweet_analysis WHERE toxicity_score IS NOT NULL")
    tox_vals = [float(r[0]) for r in cursor.fetchall() if r[0] is not None]

    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    hist = [0] * (len(bins) - 1)
    for x in tox_vals:
        for i in range(len(bins) - 1):
            if bins[i] <= x < bins[i+1]:
                hist[i] += 1
                break

    toxicity_histogram = {
        "bins": ["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"],
        "counts": hist
    }

    cursor.close()
    conn.close()

    return jsonify({
        "total": total,
        "sentiment_counts": sentiment_counts,
        "emotion_counts": emotion_counts,
        "toxicity_counts": toxicity_counts,
        "top_hashtags": top_hashtags,
        "emotion_hashtag_analysis": emotion_top_hashtags,
        "avg_toxicity_by_emotion": avg_toxicity_by_emotion,
        "hashtag_sentiment_ratio": hashtag_sentiment_ratio,
        "toxicity_histogram": toxicity_histogram
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
