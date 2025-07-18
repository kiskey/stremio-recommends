# trakt_sync.py

import os
import time
import sqlite3
import requests

# --- Configuration from Environment Variables ---
TRAKT_USERNAME = os.environ.get('TRAKT_USERNAME')
TRAKT_CLIENT_ID = os.environ.get('TRAKT_CLIENT_ID')
TRAKT_CLIENT_SECRET = os.environ.get('TRAKT_CLIENT_SECRET')
SYNC_INTERVAL = int(os.environ.get('TRAKT_SYNC_INTERVAL_MINUTES', 60)) * 60
HISTORY_DB_PATH = os.environ.get('HISTORY_DB_PATH', '/usr/src/app/persistent_data/watch_history.db')
TRAKT_API_BASE = 'https://api.trakt.tv'

def get_watched_history(media_type):
    """Fetches all watched items (movies or shows) for a user from Trakt."""
    print(f"[Trakt Sync] Fetching watched {media_type} from Trakt...")
    url = f"{TRAKT_API_BASE}/users/{TRAKT_USERNAME}/watched/{media_type}"
    headers = {
        'Content-Type': 'application/json',
        'trakt-api-version': '2',
        'trakt-api-key': TRAKT_CLIENT_ID
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        watched_ids = set()
        for item in data:
            item_key = 'movie' if media_type == 'movies' else 'show'
            if item_key in item and 'ids' in item[item_key] and item[item_key]['ids']['imdb']:
                db_type = 'movie' if media_type == 'movies' else 'tvSeries'
                watched_ids.add((item[item_key]['ids']['imdb'], db_type))
        
        print(f"[Trakt Sync] Found {len(watched_ids)} unique watched {media_type}.")
        return watched_ids
    except requests.exceptions.RequestException as e:
        print(f"[Trakt Sync] ERROR: Error fetching from Trakt API for {media_type}: {e}")
        return None

def update_local_database(watched_ids):
    """Inserts new watched items into the local SQLite database."""
    if not watched_ids:
        print("[Trakt Sync] No new items to update in the database.")
        return

    try:
        os.makedirs(os.path.dirname(HISTORY_DB_PATH), exist_ok=True)
        conn = sqlite3.connect(HISTORY_DB_PATH)
        cursor = conn.cursor()
        cursor.execute('CREATE TABLE IF NOT EXISTS history (imdb_id TEXT PRIMARY KEY, type TEXT NOT NULL, timestamp REAL NOT NULL)')
        
        update_time = time.time()
        # Use INSERT OR IGNORE to add only new items without causing errors
        cursor.executemany("INSERT OR IGNORE INTO history (imdb_id, type, timestamp) VALUES (?, ?, ?)", 
                           [(imdb_id, media_type, update_time) for imdb_id, media_type in watched_ids])
        
        conn.commit()
        conn.close()
        print(f"[Trakt Sync] Successfully updated local history database.")
    except Exception as e:
        print(f"[Trakt Sync] ERROR: Error updating local database: {e}")

def main():
    """Main loop for the sync worker."""
    if not all([TRAKT_USERNAME, TRAKT_CLIENT_ID]):
        print("[Trakt Sync] FATAL: TRAKT_USERNAME and TRAKT_CLIENT_ID environment variables must be set. Worker exiting.")
        return

    print("[Trakt Sync] Worker started.")
    while True:
        print("[Trakt Sync] --- Starting Trakt Sync Cycle ---")
        
        watched_movies = get_watched_history('movies')
        watched_shows = get_watched_history('shows')
        
        all_watched = set()
        if watched_movies:
            all_watched.update(watched_movies)
        if watched_shows:
            all_watched.update(watched_shows)
            
        if all_watched:
            update_local_database(all_watched)
        
        print(f"[Trakt Sync] --- Sync cycle complete. Sleeping for {SYNC_INTERVAL / 60:.0f} minutes. ---")
        time.sleep(SYNC_INTERVAL)

if __name__ == "__main__":
    main()
