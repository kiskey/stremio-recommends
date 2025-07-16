# main.py

import sqlite3
import pickle
import pandas as pd
from flask import Flask, jsonify, Blueprint
import time
import os
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration from Environment Variables ---
PAGE_SIZE = int(os.environ.get('PAGE_SIZE', 50))
PRIORITY_REGIONS = os.environ.get('PRIORITY_REGIONS', 'IN').split(',')
HISTORY_SEED_COUNT = 5
TOTAL_LIMIT = int(os.environ.get('TOTAL_LIMIT', 50))
ARTIFACTS_DIR = os.environ.get('ARTIFACTS_DIR', 'artifacts')
HISTORY_DB_PATH = os.path.join(ARTIFACTS_DIR, 'watch_history.db')

# --- Create a Blueprint ---
main_bp = Blueprint('main', __name__)

# --- Global variables for loaded data ---
tfidf_matrix, all_titles, indices = None, None, None

# --- Addon Manifest (Unchanged) ---
MANIFEST = {
    "id": "community.dynamic.recommendations",
    "version": "3.1.2", # Patch version for the final series fix
    "name": "For You Recommendations",
    "description": "Provides configurable, paginated, and region-sorted recommendations.",
    "types": ["movie", "series"],
    "resources": ["catalog", "meta"],
    "catalogs": [
        {"type": "movie", "id": "recs_movies", "name": "Recommended Movies", "extra": [{"name": "skip", "isRequired": False}]},
        {"type": "series", "id": "recs_series", "name": "Recommended Series", "extra": [{"name": "skip", "isRequired": False}]}
    ]
}

# --- Route Definitions ---
@main_bp.route('/manifest.json')
def manifest():
    return jsonify(MANIFEST)

# --- CORRECTED META HANDLERS ---
@main_bp.route('/meta/movie/<imdb_id>.json')
def meta_movie_logger(imdb_id):
    log_to_history(imdb_id, 'movie') # Log with internal type
    return jsonify({"err": "not found"}), 404

@main_bp.route('/meta/series/<imdb_id>.json')
def meta_series_logger(imdb_id):
    # --- FIX: Log the data source's native type ('tvSeries') ---
    log_to_history(imdb_id, 'tvSeries')
    return jsonify({"err": "not found"}), 404

# --- CORRECTED CATALOG ROUTES ---
@main_bp.route('/catalog/movie/recs_movies.json')
@main_bp.route('/catalog/movie/recs_movies/skip=<int:skip>.json')
def get_movie_recommendations(skip: int = 0):
    return generate_sorted_recommendations(media_type='movie', skip=skip)

@main_bp.route('/catalog/series/recs_series.json')
@main_bp.route('/catalog/series/recs_series/skip=<int:skip>.json')
def get_series_recommendations(skip: int = 0):
    # --- FIX: Call the helper with the data source's native type ('tvSeries') ---
    return generate_sorted_recommendations(media_type='tvSeries', skip=skip)

# --- CORE LOGIC HELPER FUNCTION ---
def generate_sorted_recommendations(media_type: str, skip: int = 0):
    conn = sqlite3.connect(HISTORY_DB_PATH)
    # The query now correctly looks for 'movie' or 'tvSeries', matching what's logged
    query = "SELECT imdb_id FROM history WHERE type = ? ORDER BY timestamp DESC LIMIT ?"
    history_df = pd.read_sql_query(query, conn, params=(media_type, HISTORY_SEED_COUNT))
    
    if history_df.empty:
        print(f"No watch history found for type '{media_type}'. Returning empty catalog.")
        conn.close()
        return jsonify({"metas": []})

    candidate_scores = {}
    full_history_ids = set(pd.read_sql_query("SELECT imdb_id FROM history", conn)['imdb_id'])
    conn.close()

    for imdb_id in history_df['imdb_id']:
        if imdb_id in indices:
            idx = indices[imdb_id]
            sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
            related_indices = sim_scores.argsort()[-31:-1][::-1]
            for rel_idx in related_indices:
                if rel_idx not in candidate_scores:
                    candidate_scores[rel_idx] = sim_scores[rel_idx]
    
    candidate_details = all_titles.iloc[list(candidate_scores.keys())].copy()
    candidate_details['score'] = candidate_details.index.map(candidate_scores)

    filtered_candidates = candidate_details[~candidate_details['tconst'].isin(full_history_ids)]
    
    # The filter is now simple and direct, as media_type is already correct ('tvSeries' or 'movie')
    typed_candidates = filtered_candidates[filtered_candidates['titleType'] == media_type]

    # Region-based sorting logic remains the same
    sorted_groups, processed_regions = [], set()
    for region in PRIORITY_REGIONS:
        region_recs = typed_candidates[typed_candidates['primary_region'] == region]
        sorted_groups.append(region_recs.sort_values(by=['score', 'averageRating'], ascending=False))
        processed_regions.add(region)
    other_recs = typed_candidates[~typed_candidates['primary_region'].isin(processed_regions)]
    sorted_groups.append(other_recs.sort_values(by=['score', 'averageRating'], ascending=False))

    if not sorted_groups: return jsonify({"metas": []})
    full_sorted_df = pd.concat(sorted_groups)
    total_limited_df = full_sorted_df.head(TOTAL_LIMIT)
    paginated_df = total_limited_df.iloc[skip : skip + PAGE_SIZE]

    # Format for Stremio response
    metas = []
    for _, row in paginated_df.iterrows():
        poster_url = f"https://images.metahub.space/poster/medium/{row['tconst']}/img"
        # Translate to the Stremio-friendly type ('series') at the very end
        output_type = 'series' if row['titleType'] == 'tvSeries' else 'movie'
        metas.append({ "id": row['tconst'], "type": output_type, "name": row['primaryTitle'], "poster": poster_url, "posterShape": "poster" })

    return jsonify({"metas": metas})

# --- Helper functions and Application Factory (Unchanged) ---
def log_to_history(imdb_id, media_type):
    try:
        os.makedirs(os.path.dirname(HISTORY_DB_PATH), exist_ok=True)
        conn = sqlite3.connect(HISTORY_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO history (imdb_id, type, timestamp) VALUES (?, ?, ?)", (imdb_id, media_type, time.time()))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error logging to history DB at {HISTORY_DB_PATH}: {e}")

def create_app():
    app = Flask(__name__)
    global tfidf_matrix, all_titles, indices
    
    required_files = [os.path.join(ARTIFACTS_DIR, f) for f in ['enriched_titles.pkl', 'tfidf_vectorizer.pkl', 'tfidf_matrix.pkl']]
    for f in required_files:
        if not os.path.exists(f): print(f"!!! FATAL ERROR: Required artifact not found in image: {f}"); exit(1)

    print("✅ Bundled artifacts found. Loading...")
    try:
        with open(os.path.join(ARTIFACTS_DIR, 'tfidf_matrix.pkl'), 'rb') as f:
            tfidf_matrix = pickle.load(f)
        all_titles = pd.read_pickle(os.path.join(ARTIFACTS_DIR, 'enriched_titles.pkl'))
        indices = pd.Series(all_titles.index, index=all_titles['tconst'])
        print("Data artifacts loaded successfully.")
    except Exception as e:
        print(f"FATAL ERROR: Failed to load artifacts: {e}"); exit(1)

    try:
        os.makedirs(os.path.dirname(HISTORY_DB_PATH), exist_ok=True)
        conn = sqlite3.connect(HISTORY_DB_PATH)
        cursor = conn.cursor()
        cursor.execute('CREATE TABLE IF NOT EXISTS history (imdb_id TEXT PRIMARY KEY, type TEXT NOT NULL, timestamp REAL NOT NULL)')
        conn.commit()
        conn.close()
        print(f"✅ History database initialized at: {HISTORY_DB_PATH}")
    except Exception as e:
        print(f"FATAL ERROR: Could not initialize history database: {e}"); exit(1)
    
    app.register_blueprint(main_bp)
    return app
