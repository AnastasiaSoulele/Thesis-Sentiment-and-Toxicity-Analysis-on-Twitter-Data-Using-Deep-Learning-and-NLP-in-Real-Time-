# init_db.py
from db import get_db_connection

schema = """
DROP TABLE IF EXISTS tweet_analysis;

CREATE TABLE tweet_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tweet_id TEXT,
    username TEXT,
    tweet_text TEXT,
    sentiment_label TEXT,
    sentiment_score REAL,
    emotion_label TEXT,
    toxicity_level TEXT,
    toxicity_score REAL,
    hashtags TEXT,
    analyzed_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

conn = get_db_connection()
cursor = conn.cursor()
cursor.executescript(schema)  
conn.commit()
cursor.close()
conn.close()


print("SQLite table created successfully.")
