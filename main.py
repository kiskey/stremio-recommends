# main.py

import sqlite3
import pickle
import pandas as pd
from flask import Flask, jsonify
import time
import os
from sklearn.metrics.pairwise import cosine_similarity # We now need this here

# --- APPLICATION SETUP (unchanged) ---
app = Flask(__name__)
RECOMMENDATIONS_LIMIT = 50
HISTORY_SEED_COUNT = 5
ARTIFACTS_DIR = os.environ.get('ARTIFACTS_DIR', 'artifacts')
HISTORY_DB_PATH = os.path.join(ARTIFACTS_DIR, 'watch_history.db')

# --- LOAD NEW ARTIFACTS AT STARTUP ---
try:
    print(f"Loading data artifacts from: {ARTIFACTS_DIR}")
    # Load the three key components of our "Split Brain" model
    with open(os.path.join(ARTIFACTS_DIR, 'tfidf_vectorizer.pkl'), 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'tfidf_matrix.pkl'), 'rb') as f:
        tfidf_matrix = pickle.load(f)
    
    all_titles = pd.read_pickle(os.path.join(ARTIFACTS_DIR, 'enriched_titles.pkl'))
    indices = pd.Series(all_titles.index, index=all_titles['tconst'])
    print("Data artifacts loaded successfully.")
except FileNotFoundError:
    print(f"FATAL ERROR: Data artifacts not found in '{ARTIFACTS_DIR}'.")
    exit()

# --- DB SETUP and MANIFEST (unchanged) ---
# ... (init_history_db, log_to_history, MANIFEST, and meta loggers are all identical) ...
def init_history_db():
    conn = sqlite3.connect(HISTORY_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS history (imdb_id TEXT PRIMARY KEY, type TEXT NOT NULL, timestamp REAL NOT NULL)')
    conn.commit()
    conn.close()

# --- REVISED RECOMMENDATION ENDPOINT ---
@app.route('/catalog/movie/for_you_recs.json')
@app.route('/catalog/series/for_you_recs.json')
def get_recommendations_catalog():
    conn = sqlite3.connect(HISTORY_DB_PATH)
    history_df = pd.read_sql_query(f"SELECT imdb_id FROM history ORDER BY timestamp DESC LIMIT {HISTORY_SEED_COUNT}", conn)
    
    if history_df.empty:
        conn.close()
        return jsonify({"metas": []})

    candidate_scores = {} # Use a dict to store scores to avoid duplicates and keep the best score
    full_history_ids = set(pd.read_sql_query("SELECT imdb_id FROM history", conn)['imdb_id'])
    conn.close()

    for imdb_id in history_df['imdb_id']:
        if imdb_id in indices:
            idx = indices[imdb_id]
            
            # --- RUNTIME CALCULATION ---
            # Calculate similarity between this ONE item and ALL others.
            # This is fast and uses very little memory.
            sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
            
            # Get the indices of the top 20 items
            related_indices = sim_scores.argsort()[-21:-1][::-1]

            for rel_idx in related_indices:
                if rel_idx not in candidate_scores: # Add new candidates
                    candidate_scores[rel_idx] = sim_scores[rel_idx]
    
    # Convert candidates to a DataFrame for filtering
    candidate_indices = list(candidate_scores.keys())
    candidate_details = all_titles.iloc[candidate_indices].copy()
    candidate_details['score'] = candidate_details.index.map(candidate_scores)

    filtered_candidates = candidate_details[~candidate_details['tconst'].isin(full_history_ids)]

    indian_recs = filtered_candidates[filtered_candidates['is_indian'] == True]
    international_recs = filtered_candidates[filtered_candidates['is_indian'] == False]
    
    # Sort by a mix of similarity score and rating for better results
    sorted_indian = indian_recs.sort_values(by=['score', 'averageRating'], ascending=False)
    sorted_international = international_recs.sort_values(by=['score', 'averageRating'], ascending=False)

    final_recs_df = pd.concat([sorted_indian, sorted_international]).head(RECOMMENDATIONS_LIMIT)

    metas = []
    for _, row in final_recs_df.iterrows():
        metas.append({"id": row['tconst'], "type": row['titleType'], "name": row['primaryTitle'], "poster": None, "posterShape": "poster"})

    return jsonify({"metas": metas})


if __name__ == '__main__':
    init_history_db()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
