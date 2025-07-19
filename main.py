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
HISTORY_SEED_COUNT = int(os.environ.get('HISTORY_SEED_COUNT', 5))
TOTAL_LIMIT = int(os.environ.get('TOTAL_LIMIT', 50))
MINIMUM_RATING = float(os.environ.get('MINIMUM_RATING', 4.9))
ARTIFACTS_DIR = os.environ.get('ARTIFACTS_DIR', 'artifacts')
HISTORY_DB_PATH = os.environ.get('HISTORY_DB_PATH', '/usr/src/app/persistent_data/watch_history.db')

# --- Create a Blueprint ---
main_bp = Blueprint('main', __name__)

# --- Global variables for loaded data ---
tfidf_matrix = None
all_titles = None
indices = None

# --- Addon Manifest ---
MANIFEST = {
    "id": "community.dynamic.recommendations",
    "version": "4.2.2", # Final stable version
    "name": "For You Recommendations (Trakt Synced)",
    "description": "Uses your Trakt.tv watch history to provide accurate recommendations.",
    "types": ["movie", "series"],
    "resources": ["catalog"], # We only provide catalogs, not metadata
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

# --- Catalog Routes with Pagination ---
@main_bp.route('/catalog/movie/recs_movies.json')
@main_bp.route('/catalog/movie/recs_movies/skip=<int:skip>.json')
def get_movie_recommendations(skip: int = 0):
    """Endpoint for the 'Recommended Movies' catalog."""
    return generate_sorted_recommendations(media_type='movie', skip=skip)

@main_bp.route('/catalog/series/recs_series.json')
@main_bp.route('/catalog/series/recs_series/skip=<int:skip>.json')
def get_series_recommendations(skip: int = 0):
    """Endpoint for the 'Recommended Series' catalog."""
    return generate_sorted_recommendations(media_type='tvSeries', skip=skip)

# --- CORE LOGIC HELPER FUNCTION ---
def generate_sorted_recommendations(media_type: str, skip: int = 0):
    """
    Generates a dedicated pool of recommendations for a specific media type,
    then filters, sorts, and paginates them.
    """
    try:
        conn = sqlite3.connect(HISTORY_DB_PATH)
        query = "SELECT imdb_id FROM history WHERE type = ? ORDER BY timestamp DESC LIMIT ?"
        history_df = pd.read_sql_query(query, conn, params=(media_type, HISTORY_SEED_COUNT))
        if history_df.empty:
            conn.close()
            return jsonify({"metas": []})
        full_history_ids = set(pd.read_sql_query("SELECT imdb_id FROM history", conn)['imdb_id'])
        conn.close()
    except Exception as e:
        print(f"[main.py] ERROR: Could not read from history database: {e}")
        return jsonify({"metas": []})

    # Independent Candidate Generation Loop
    candidate_ids = set()
    # Aim for a slightly larger pool to account for subsequent filtering
    candidate_pool_target_size = TOTAL_LIMIT + len(history_df) + 100 
    for imdb_id in history_df['imdb_id']:
        if len(candidate_ids) >= candidate_pool_target_size: break
        if imdb_id in indices:
            idx = indices[imdb_id]
            sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
            for related_idx in sim_scores.argsort()[::-1]:
                if len(candidate_ids) >= candidate_pool_target_size: break
                rec_details = all_titles.iloc[related_idx]
                rec_id = rec_details['tconst']
                # Generate a mixed pool first, we will filter by type later
                if rec_id not in full_history_ids and rec_id not in candidate_ids:
                    candidate_ids.add(rec_id)
    
    if not candidate_ids: return jsonify({"metas": []})

    candidate_details = all_titles[all_titles['tconst'].isin(list(candidate_ids))].copy()
    
    # --- Corrected Logic Flow ---
    # 1. First, filter for the correct media type (movie or tvSeries).
    typed_candidates = candidate_details[candidate_details['titleType'] == media_type]

    # 2. Now, apply the conditional rating filter to this clean, type-specific pool.
    indian_candidates = typed_candidates[typed_candidates['primary_region'] == 'IN']
    non_indian_candidates = typed_candidates[typed_candidates['primary_region'] != 'IN']
    
    # Keep Indian content if rating is high enough OR if rating is not available (0.0).
    filtered_indian = indian_candidates[
        (indian_candidates['averageRating'] >= MINIMUM_RATING) |
        (indian_candidates['averageRating'] == 0.0)
    ]
    # Apply the strict filter to all other content.
    filtered_non_indian = non_indian_candidates[non_indian_candidates['averageRating'] >= MINIMUM_RATING]
    
    rated_candidates = pd.concat([filtered_indian, filtered_non_indian])
    
    # Region-based sorting logic
    sorted_groups, processed_regions = [], set()
    for region in PRIORITY_REGIONS:
        region_recs = rated_candidates[rated_candidates['primary_region'] == region]
        sorted_groups.append(region_recs.sort_values(by=['averageRating'], ascending=False))
        processed_regions.add(region)
    other_recs = rated_candidates[~rated_candidates['primary_region'].isin(processed_regions)]
    sorted_groups.append(other_recs.sort_values(by=['averageRating'], ascending=False))

    full_sorted_df = pd.concat(sorted_groups)
    
    # Apply total limit and pagination
    total_limited_df = full_sorted_df.head(TOTAL_LIMIT)
    paginated_df = total_limited_df.iloc[skip : skip + PAGE_SIZE]

    # Calculate 'hasMore' correctly for pagination
    num_total_results = len(total_limited_df)
    has_more = (skip + PAGE_SIZE) < num_total_results

    # Format for Stremio response
    metas = []
    for _, row in paginated_df.iterrows():
        poster_url = f"https://images.metahub.space/poster/medium/{row['tconst']}/img"
        output_type = 'series' if row['titleType'] == 'tvSeries' else 'movie'
        metas.append({
            "id": row['tconst'], "type": output_type, "name": row['primaryTitle'],
            "poster": poster_url, "posterShape": "poster"
        })

    return jsonify({"metas": metas, "hasMore": has_more})

# --- Application Factory Function ---
def create_app():
    """Creates and configures the Flask application."""
    app = Flask(__name__)
    global tfidf_matrix, all_titles, indices
    
    # Pre-flight check is now handled by the entrypoint.sh script.
    print("[main.py] Loading bundled artifacts...")
    try:
        with open(os.path.join(ARTIFACTS_DIR, 'tfidf_matrix.pkl'), 'rb') as f:
            tfidf_matrix = pickle.load(f)
        all_titles = pd.read_pickle(os.path.join(ARTIFACTS_DIR, 'enriched_titles.pkl'))
        indices = pd.Series(all_titles.index, index=all_titles['tconst'])
        print("[main.py] ✅ Data artifacts loaded successfully.")
    except Exception as e:
        print(f"[main.py] FATAL ERROR: Failed to load artifacts: {e}"); exit(1)

    # Initialize history DB
    try:
        os.makedirs(os.path.dirname(HISTORY_DB_PATH), exist_ok=True)
        conn = sqlite3.connect(HISTORY_DB_PATH)
        cursor = conn.cursor()
        cursor.execute('CREATE TABLE IF NOT EXISTS history (imdb_id TEXT PRIMARY KEY, type TEXT NOT NULL, timestamp REAL NOT NULL)')
        conn.commit()
        conn.close()
        print(f"[main.py] ✅ History database initialized at: {HISTORY_DB_PATH}")
    except Exception as e:
        print(f"[main.py] FATAL ERROR: Could not initialize history database: {e}"); exit(1)
    
    app.register_blueprint(main_bp)
    return app
