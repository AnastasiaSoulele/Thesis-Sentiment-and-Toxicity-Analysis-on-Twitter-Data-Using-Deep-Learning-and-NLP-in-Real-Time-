import sqlite3
import pandas as pd

# Σύνδεση με SQLite βάση
conn = sqlite3.connect("sentimentx.db")

# 1. Εμφάνιση πλήθους tweets
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM tweet_analysis")
count = cursor.fetchone()[0]
print(f"Tweets στη βάση μέχρι τώρα: {count}")

# 2. Δείγμα 10 tweets
df = pd.read_sql_query("SELECT * FROM tweet_analysis LIMIT 10", conn)
print("\nΔείγμα εγγραφών:")
print(df)

# 3. Εμφάνιση μοναδικών συναισθημάτων
cursor.execute("SELECT emotion_label, COUNT(*) FROM tweet_analysis GROUP BY emotion_label")
emotion_stats = cursor.fetchall()

print("\nΣυχνότητα ανά συναίσθημα (emotion_label):")
for emotion, cnt in emotion_stats:
    print(f"- {emotion}: {cnt} tweets")


emotion_df = pd.DataFrame(emotion_stats, columns=["Emotion", "Count"])
print("\nΩς DataFrame:")
print(emotion_df)

conn.close()
