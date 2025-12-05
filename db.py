# db.py
import sqlite3

def get_db_connection():
    conn = sqlite3.connect("sentimentx.db")  
    conn.row_factory = sqlite3.Row          # Για να επιστρέφει αποτελέσματα ως dictionaries
    return conn

print("SQLite connection established.")
