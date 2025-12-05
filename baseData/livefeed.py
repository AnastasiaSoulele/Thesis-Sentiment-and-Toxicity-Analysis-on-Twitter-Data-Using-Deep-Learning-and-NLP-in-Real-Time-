from datasets import load_dataset
import json
import random
import os

# 1. Ορισμός των υποσυνόλων 
subsets = ["sentiment", "emotion", "hate", "offensive", "irony"]

# 2. Λίστα για όλα τα tweets
all_data = []

# 3. Φόρτωμα και ένωση των datasets
for subset in subsets:
    print(f"Φόρτωση: {subset}")
    try:
        ds = load_dataset("tweet_eval", subset)
        combined = list(ds["train"]) + list(ds["validation"]) + list(ds["test"])
        for item in combined:
            all_data.append({
                "text": item["text"],
                "label": item.get("label", -1),  
                "subset": subset
            })
    except Exception as e:
        print(f"Πρόβλημα στο {subset}: {e}")

# 4. Τυχαιοποίηση όλων των tweets
random.shuffle(all_data)

# 5. Δημιουργία φακέλου για batches
os.makedirs("fake_batches", exist_ok=True)

# 6. Δημιουργία batches
batch_size = 50
for i in range(0, len(all_data), batch_size):
    batch = all_data[i:i + batch_size]
    tweets = []
    for j, item in enumerate(batch):
        tweets.append({
            "id": f"{i + j + 1}",
            "text": item["text"],
            "username": f"user_{random.randint(1000, 9999)}",
            "source": item["subset"]  
        })

    with open(f"fake_batches/batch_{i // batch_size + 1}.json", "w", encoding="utf-8") as f:
        json.dump(tweets, f, ensure_ascii=False, indent=2)

print("Δημιουργήθηκαν batches στον φάκελο: fake_batches/")

