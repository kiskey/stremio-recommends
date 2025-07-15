# main.py

import sqlite3
import pickle
import pandas as pd
from flask import Flask, jsonify
import time
import os

# --- APPLICATION SETUP ---
app = Flask(__name__)
RECOMMENDATIONS_LIMIT = 50
HISTORY_SEED_COUNT = 5
ARTIFACTS_DIR = os.environ.get('ARTIFACTS_DIR', 'artifacts')
HISTORY_DB_PATH = os.path.join(ARTIFACTS_DIR, 'watch_history.db')

# --- LOAD DATA ARTIFACTS AT STARTUP ---
try:
    print(f"Loading data artifacts from: {ARTIFACTS_DIR}")
    with open(os.path.join(ARTIFACTS_DIR, 'cosine_sim.pkl'), 'rb') as f:
        cosine_sim = pickle.load(f)
    
    all_titles = pd.read_pickle(os.path.join(ARTIFACTS_DIR, 'enriched_titles.pkl'))
    indices = pd.Series(all_titles.index, index=all_titles['tconst'])
    print("Data artifacts loaded successfully.")
except FileNotFoundError:
    print(f"FATAL ERROR: Data artifacts not found in '{ARTIFACTS_DIR}'.")
    print("Please ensure the artifacts volume is correctly mounted.")
    exit()

# --- DATABASE SETUP FOR HISTORY ---
def init_history_db():
    conn = sqlite3.connect(HISTORY_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            imdb_id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            timestamp REAL NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def log_to_history(imdb_id, media_type):
    conn = sqlite3.connect(HISTORY_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO history (imdb_id, type, timestamp) VALUES (?, ?, ?)",
        (imdb_id, media_type, time.time())
    )
    conn.commit()
    conn.close()

# --- MANIFEST ---
MANIFEST = {
    "id": "community.dynamic.recommendations",
    "version": "1.1.0",
    "name": "For You Recommendations",
    "description": "A dynamic catalog of recommendations based on your viewing history.",
    "types": ["movie", "series"],
    "resources": ["catalog", "meta"],
    "catalogs": [{
        "type": "movie",
        "id": "for_you_recs",
        "name": "For You"
    }]
}

# --- ENDPOINTS ---
@app.route('/manifest.json')
def manifest():
    return jsonify(MANIFEST)

@app.route('/meta/movie/<imdb_id>.json')
def meta_movie_logger(imdb_id):
    log_to_history(imdb_id, 'movie')
    return jsonify({"meta": {}})

@app.route('/meta/series/<imdb_id>.json')
def meta_series_logger(imdb_id):
    log_to_history(imdb_id, 'series')
    return jsonify({"meta": {}})

@app.route('/catalog/movie/for_you_recs.json')
@app.route('/catalog/series/for_you_recs.json')
def get_recommendations_catalog():
    conn = sqlite3.connect(HISTORY_DB_PATH)
    history_df = pd.read_sql_query(f"SELECT imdb_id FROM history ORDER BY timestamp DESC LIMIT {HISTORY_SEED_COUNT}", conn)
    
    if history_df.empty:
        conn.close()
        return jsonify({"metas": []})

    candidate_indices = set()
    full_history_ids = set(pd.read_sql_query("SELECT imdb_id FROM history", conn)['imdb_id'])
    conn.close()

    for imdb_id in history_df['imdb_id']:
        if imdb_id in indices:
            idx = indices[imdb_id]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            top_indices = {i[0] for i in sim_scores[1:21]}
            candidate_indices.update(top_indices)

    candidate_details = all_titles.iloc[list(candidate_indices)]
    filtered_candidates = candidate_details[~candidate_details['tconst'].isin(full_history_ids)]

    indian_recs = filtered_candidates[filtered_candidates['is_indian'] == True]
    international_recs = filtered_candidates[filtered_candidates['is_indian'] == False]
    
    sorted_indian = indian_recs.sort_values(by='averageRating', ascending=False)
    sorted_international = international_recs.sort_values(by='averageRating', ascending=False)

    final_recs_df = pd.concat([sorted_indian, sorted_international]).head(RECOMMENDATIONS_LIMIT)

    metas = []
    for _, row in final_recs_df.iterrows():
        metas.append({
            "id": row['tconst'],
            "type": row['titleType'],
            "name": row['primaryTitle'],
            "poster": None,
            "posterShape": "poster"
        })

    return jsonify({"metas": metas})

if __name__ == '__main__':
    init_history_db()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
