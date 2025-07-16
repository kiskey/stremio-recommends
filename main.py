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
tfidf_matrix = None
all_titles = None
indices = None

# --- Addon Manifest (with Pagination) ---
MANIFEST = {
    "id": "community.dynamic.recommendations",
    "version": "3.2.0", # Version bump for new recommendation architecture
    "name": "For You Recommendations",
    "description": "Provides robust, independent, and configurable recommendations.",
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
    """Provides the addon's manifest to Stremio."""
    return jsonify(MANIFEST)

@main_bp.route('/meta/movie/<imdb_id>.json')
def meta_movie_logger(imdb_id):
    """Logs history and then "redirects" to other meta addons by returning a 404."""
    log_to_history(imdb_id, 'movie')
    return jsonify({"err": "not found"}), 404

@main_bp.route('/meta/series/<imdb_id>.json')
def meta_series_logger(imdb_id):
    """Logs history and then "redirects" to other meta addons by returning a 404."""
    log_to_history(imdb_id, 'tvSeries') # Use internal type
    return jsonify({"err": "not found"}), 404

# --- Catalog Routes with Pagination ---

@main_bp.route('/catalog/movie/recs_movies.json')
@main_bp.route('/catalog/movie/recs_movies/skip=<int:skip>.json')
def get_movie_recommendations(skip: int = 0):
    """Endpoint for the 'Recommended Movies' catalog with pagination."""
    return generate_sorted_recommendations(media_type='movie', skip=skip)

@main_bp.route('/catalog/series/recs_series.json')
@main_bp.route('/catalog/series/recs_series/skip=<int:skip>.json')
def get_series_recommendations(skip: int = 0):
    """Endpoint for the 'Recommended Series' catalog with pagination."""
    return generate_sorted_recommendations(media_type='tvSeries', skip=skip)

# --- CORE LOGIC HELPER FUNCTION ---

def generate_sorted_recommendations(media_type: str, skip: int = 0):
    """
    Generates a dedicated pool of recommendations for a specific media type,
    then sorts and paginates them.
    """
    conn = sqlite3.connect(HISTORY_DB_PATH)
    query = "SELECT imdb_id FROM history WHERE type = ? ORDER BY timestamp DESC LIMIT ?"
    history_df = pd.read_sql_query(query, conn, params=(media_type, HISTORY_SEED_COUNT))
    
    if history_df.empty:
        conn.close()
        return jsonify({"metas": []})

    full_history_ids = set(pd.read_sql_query("SELECT imdb_id FROM history", conn)['imdb_id'])
    conn.close()

    # --- NEW: Independent Candidate Generation Loop ---
    # This loop ensures we gather enough candidates OF THE CORRECT TYPE.
    candidate_ids = set()
    candidate_pool_target_size = TOTAL_LIMIT + len(history_df) # Aim for a slightly larger pool

    for imdb_id in history_df['imdb_id']:
        if len(candidate_ids) >= candidate_pool_target_size:
            break # Stop if we have enough candidates

        if imdb_id in indices:
            idx = indices[imdb_id]
            sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
            
            # Iterate through the full sorted list of similar items
            for related_idx in sim_scores.argsort()[::-1]:
                # Stop processing this seed if we have enough candidates
                if len(candidate_ids) >= candidate_pool_target_size:
                    break

                # Get details of the potential recommendation
                rec_details = all_titles.iloc[related_idx]
                rec_id = rec_details['tconst']
                rec_type = rec_details['titleType']

                # Filter *before* adding to the pool to ensure relevance
                if rec_type == media_type and rec_id not in full_history_ids and rec_id not in candidate_ids:
                    candidate_ids.add(rec_id)
    
    if not candidate_ids:
        return jsonify({"metas": []})

    # Now we have a clean pool of candidates of the correct type
    candidate_details = all_titles[all_titles['tconst'].isin(list(candidate_ids))].copy()
    
    # Region-based sorting logic
    sorted_groups, processed_regions = [], set()
    for region in PRIORITY_REGIONS:
        region_recs = candidate_details[candidate_details['primary_region'] == region]
        sorted_groups.append(region_recs.sort_values(by=['averageRating'], ascending=False)) # Sort by rating
        processed_regions.add(region)
    other_recs = candidate_details[~candidate_details['primary_region'].isin(processed_regions)]
    sorted_groups.append(other_recs.sort_values(by=['averageRating'], ascending=False))

    full_sorted_df = pd.concat(sorted_groups)
    
    # Apply total limit and pagination
    total_limited_df = full_sorted_df.head(TOTAL_LIMIT)
    paginated_df = total_limited_df.iloc[skip : skip + PAGE_SIZE]

    # Calculate 'hasMore' correctly
    num_total_results = len(total_limited_df)
    has_more = (skip + PAGE_SIZE) < num_total_results

    # Format for Stremio response
    metas = []
    for _, row in paginated_df.iterrows():
        poster_url = f"https://images.metahub.space/poster/medium/{row['tconst']}/img"
        output_type = 'series' if row['titleType'] == 'tvSeries' else 'movie'
        metas.append({
            "id": row['tconst'],
            "type": output_type,
            "name": row['primaryTitle'],
            "poster": poster_url,
            "posterShape": "poster"
        })

    return jsonify({"metas": metas, "hasMore": has_more})

# --- Helper functions ---

def log_to_history(imdb_id, media_type):
    """Writes a viewed item to the persistent history database."""
    try:
        os.makedirs(os.path.dirname(HISTORY_DB_PATH), exist_ok=True)
        conn = sqlite3.connect(HISTORY_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO history (imdb_id, type, timestamp) VALUES (?, ?, ?)", (imdb_id, media_type, time.time()))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error logging to history DB at {HISTORY_DB_PATH}: {e}")

# --- Application Factory Function ---

def create_app():
    """Creates and configures the Flask application."""
    app = Flask(__name__)
    
    global tfidf_matrix, all_titles, indices
    
    # Pre-flight checks for critical files
    required_files = [os.path.join(ARTIFACTS_DIR, f) for f in ['enriched_titles.pkl', 'tfidf_vectorizer.pkl', 'tfidf_matrix.pkl']]
    for f in required_files:
        if not os.path.exists(f):
            print(f"!!! FATAL ERROR: Required artifact not found in image: {f}"); exit(1)

    print("✅ Bundled artifacts found. Loading...")
    try:
        with open(os.path.join(ARTIFACTS_DIR, 'tfidf_matrix.pkl'), 'rb') as f:
            tfidf_matrix = pickle.load(f)
        all_titles = pd.read_pickle(os.path.join(ARTIFACTS_DIR, 'enriched_titles.pkl'))
        indices = pd.Series(all_titles.index, index=all_titles['tconst'])
        print("Data artifacts loaded successfully.")
    except Exception as e:
        print(f"FATAL ERROR: Failed to load artifacts: {e}"); exit(1)

    # Initialize history DB
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
