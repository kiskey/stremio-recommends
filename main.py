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
main_bp = Blueprint('main', __name__)

# --- Global variables for loaded data ---
tfidf_matrix = None
all_titles = None
indices = None

# --- Addon Manifest (UPDATED FOR TWO CATALOGS) ---
MANIFEST = {
    "id": "community.dynamic.recommendations",
    "version": "2.0.0", # Major version bump for new features
    "name": "For You Recommendations",
    "description": "Provides separate, personalized catalogs for movies and series based on your viewing history.",
    "types": ["movie", "series"],
    "resources": ["catalog", "meta"],
    "catalogs": [
        {
            "type": "movie",
            "id": "recs_movies",
            "name": "Recommended Movies"
        },
        {
            "type": "series",
            "id": "recs_series",
            "name": "Recommended Series"
        }
    ]
}

# --- Route Definitions ---

@main_bp.route('/manifest.json')
def manifest():
    return jsonify(MANIFEST)

# Meta loggers remain the same
@main_bp.route('/meta/movie/<imdb_id>.json')
def meta_movie_logger(imdb_id):
    log_to_history(imdb_id, 'movie')
    return jsonify({"meta": {}})

@main_bp.route('/meta/series/<imdb_id>.json')
def meta_series_logger(imdb_id):
    log_to_history(imdb_id, 'series')
    return jsonify({"meta": {}})

# --- NEW SEPARATE CATALOG ROUTES ---

@main_bp.route('/catalog/movie/recs_movies.json')
def get_movie_recommendations():
    """Endpoint for the 'Recommended Movies' catalog."""
    return generate_sorted_recommendations(media_type='movie')

@main_bp.route('/catalog/series/recs_series.json')
def get_series_recommendations():
    """Endpoint for the 'Recommended Series' catalog."""
    return generate_sorted_recommendations(media_type='series')

# --- CORE LOGIC HELPER FUNCTION ---

def generate_sorted_recommendations(media_type: str):
    """
    Generates a pool of recommendations, then filters and sorts them
    based on the specified media type and user requirements.
    """
    # 1. Generate a common pool of candidates from history
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
            related_indices = sim_scores.argsort()[-31:-1][::-1] # Get a slightly larger pool (top 30)
            for rel_idx in related_indices:
                if rel_idx not in candidate_scores:
                    candidate_scores[rel_idx] = sim_scores[rel_idx]
    
    candidate_indices = list(candidate_scores.keys())
    candidate_details = all_titles.iloc[candidate_indices].copy()
    candidate_details['score'] = candidate_details.index.map(candidate_scores)

    # 2. Filter out already watched items
    filtered_candidates = candidate_details[~candidate_details['tconst'].isin(full_history_ids)]

    # 3. Filter for the specific media type (movie or series)
    typed_candidates = filtered_candidates[filtered_candidates['titleType'] == media_type]

    # 4. Separate by region
    indian_recs = typed_candidates[typed_candidates['is_indian'] == True]
    international_recs = typed_candidates[typed_candidates['is_indian'] == False]
    
    # 5. Sort each regional group by relevance (score) and rating
    sorted_indian = indian_recs.sort_values(by=['score', 'averageRating'], ascending=False)
    sorted_international = international_recs.sort_values(by=['score', 'averageRating'], ascending=False)

    # 6. Combine (Indian first) and limit to 50
    final_recs_df = pd.concat([sorted_indian, sorted_international]).head(RECOMMENDATIONS_LIMIT)

    # 7. Format for Stremio response
    metas = []
    for _, row in final_recs_df.iterrows():
        poster_url = f"https://images.metahub.space/poster/medium/{row['tconst']}/img"
        metas.append({
            "id": row['tconst'],
            "type": row['titleType'],
            "name": row['primaryTitle'],
            "poster": poster_url,
            "posterShape": "poster"
        })

    return jsonify({"metas": metas})

# --- Helper functions and Application Factory (unchanged) ---

def log_to_history(imdb_id, media_type):
    # ... (implementation is identical to previous version)
    try:
        conn = sqlite3.connect(HISTORY_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO history (imdb_id, type, timestamp) VALUES (?, ?, ?)", (imdb_id, media_type, time.time()))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error logging to history DB: {e}")

def create_app():
    # ... (implementation is identical to previous version)
    app = Flask(__name__)
    global tfidf_matrix, all_titles, indices
    
    print(f"Loading data artifacts from: {ARTIFACTS_DIR}")
    try:
        with open(os.path.join(ARTIFACTS_DIR, 'tfidf_matrix.pkl'), 'rb') as f:
            tfidf_matrix = pickle.load(f)
        all_titles = pd.read_pickle(os.path.join(ARTIFACTS_DIR, 'enriched_titles.pkl'))
        indices = pd.Series(all_titles.index, index=all_titles['tconst'])
        print("Data artifacts loaded successfully.")
    except FileNotFoundError:
        print(f"FATAL ERROR: Data artifacts not found in '{ARTIFACTS_DIR}'. Cannot start app.")
        exit(1)
    
    try:
        conn = sqlite3.connect(HISTORY_DB_PATH)
        cursor = conn.cursor()
        cursor.execute('CREATE TABLE IF NOT EXISTS history (imdb_id TEXT PRIMARY KEY, type TEXT NOT NULL, timestamp REAL NOT NULL)')
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"FATAL ERROR: Could not initialize history database at {HISTORY_DB_PATH}: {e}")
        exit(1)
    
    app.register_blueprint(main_bp)
    return app
