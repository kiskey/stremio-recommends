# main.py

import sqlite3
import pickle
import pandas as pd
from flask import Flask, jsonify, Blueprint
import time
import os
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
RECOMMENDATIONS_LIMIT = 50
HISTORY_SEED_COUNT = 5
ARTIFACTS_DIR = os.environ.get('ARTIFACTS_DIR', 'artifacts')
HISTORY_DB_PATH = os.path.join(ARTIFACTS_DIR, 'watch_history.db')

# --- Create a Blueprint ---
# A Blueprint is a way to organize a group of related views and other code.
# Instead of registering views and other code directly with an application,
# they are registered with a blueprint.
main_bp = Blueprint('main', __name__)

# --- Load Data Artifacts (will be loaded inside the factory) ---
# We define them globally as None first
tfidf_matrix = None
all_titles = None
indices = None

# --- MANIFEST (Defined as a simple dictionary) ---
MANIFEST = {
    "id": "community.dynamic.recommendations",
    "version": "1.2.0", # Version bump
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

# --- Route Definitions (Now attached to the Blueprint) ---
@main_bp.route('/manifest.json')
def manifest():
    return jsonify(MANIFEST)

@main_bp.route('/meta/movie/<imdb_id>.json')
def meta_movie_logger(imdb_id):
    log_to_history(imdb_id, 'movie')
    return jsonify({"meta": {}})

@main_bp.route('/meta/series/<imdb_id>.json')
def meta_series_logger(imdb_id):
    log_to_history(imdb_id, 'series')
    return jsonify({"meta": {}})

@main_bp.route('/catalog/movie/for_you_recs.json')
@main_bp.route('/catalog/series/for_you_recs.json')
def get_recommendations_catalog():
    conn = sqlite3.connect(HISTORY_DB_PATH)
    history_df = pd.read_sql_query(f"SELECT imdb_id FROM history ORDER BY timestamp DESC LIMIT {HISTORY_SEED_COUNT}", conn)
    
    if history_df.empty:
        conn.close()
        return jsonify({"metas": []})

    candidate_scores = {}
    full_history_ids = set(pd.read_sql_query("SELECT imdb_id FROM history", conn)['imdb_id'])
    conn.close()

    for imdb_id in history_df['imdb_id']:
        if imdb_id in indices:
            idx = indices[imdb_id]
            sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
            related_indices = sim_scores.argsort()[-21:-1][::-1]
            for rel_idx in related_indices:
                if rel_idx not in candidate_scores:
                    candidate_scores[rel_idx] = sim_scores[rel_idx]
    
    candidate_indices = list(candidate_scores.keys())
    candidate_details = all_titles.iloc[candidate_indices].copy()
    candidate_details['score'] = candidate_details.index.map(candidate_scores)

    filtered_candidates = candidate_details[~candidate_details['tconst'].isin(full_history_ids)]

    indian_recs = filtered_candidates[filtered_candidates['is_indian'] == True]
    international_recs = filtered_candidates[filtered_candidates['is_indian'] == False]
    
    sorted_indian = indian_recs.sort_values(by=['score', 'averageRating'], ascending=False)
    sorted_international = international_recs.sort_values(by=['score', 'averageRating'], ascending=False)

    final_recs_df = pd.concat([sorted_indian, sorted_international]).head(RECOMMENDATIONS_LIMIT)

    metas = []
    for _, row in final_recs_df.iterrows():
        metas.append({"id": row['tconst'], "type": row['titleType'], "name": row['primaryTitle'], "poster": None, "posterShape": "poster"})

    return jsonify({"metas": metas})

# --- Helper functions ---
def log_to_history(imdb_id, media_type):
    conn = sqlite3.connect(HISTORY_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO history (imdb_id, type, timestamp) VALUES (?, ?, ?)", (imdb_id, media_type, time.time()))
    conn.commit()
    conn.close()

# --- Application Factory Function ---
def create_app():
    # This is the factory. It creates and configures the app.
    app = Flask(__name__)
    
    # Use 'global' to modify the variables defined outside this function
    global tfidf_matrix, all_titles, indices
    
    # Load the data artifacts within the app context
    print(f"Loading data artifacts from: {ARTIFACTS_DIR}")
    try:
        with open(os.path.join(ARTIFACTS_DIR, 'tfidf_matrix.pkl'), 'rb') as f:
            tfidf_matrix = pickle.load(f)
        all_titles = pd.read_pickle(os.path.join(ARTIFACTS_DIR, 'enriched_titles.pkl'))
        indices = pd.Series(all_titles.index, index=all_titles['tconst'])
        print("Data artifacts loaded successfully.")
    except FileNotFoundError:
        print(f"FATAL ERROR: Data artifacts not found in '{ARTIFACTS_DIR}'.")
        # In a real app, you might raise an exception or handle this differently
        exit()

    # Create the history database if it doesn't exist
    conn = sqlite3.connect(HISTORY_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS history (imdb_id TEXT PRIMARY KEY, type TEXT NOT NULL, timestamp REAL NOT NULL)')
    conn.commit()
    conn.close()
    
    # Register the blueprint with the app. This is where the routes are connected.
    app.register_blueprint(main_bp)
    
    return app
