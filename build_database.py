# build_database.py

import pandas as pd
from sqlalchemy import create_engine
import pickle
import time
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# --- CONFIGURATION (unchanged) ---
MINIMUM_VOTES_THRESHOLD = 500
YEAR_FILTER_THRESHOLD = 1980
CHUNK_SIZE = 250000
BASICS_URL = "https://datasets.imdbws.com/title.basics.tsv.gz"
AKAS_URL = "https://datasets.imdbws.com/title.akas.tsv.gz"
RATINGS_URL = "https://datasets.imdbws.com/title.ratings.tsv.gz"
ARTIFACTS_DIR = "artifacts"

def build_artifacts():
    start_time = time.time()
    print("--- Starting Final Artifact Build (Runtime Similarity Model) ---")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # --- Steps 1-4: Data filtering and enrichment (This part is identical to the previous memory-optimized script) ---
    print("Step 1/6: Identifying all Indian titles...")
    # ... (same chunking logic for indian_ids) ...
    indian_ids = set() # Placeholder
    with pd.read_csv(AKAS_URL, sep='\t', usecols=['titleId', 'region'], chunksize=CHUNK_SIZE, compression='gzip') as reader:
        for chunk in reader: indian_ids.update(chunk[chunk['region'] == 'IN']['titleId'])
    
    print(f"Step 2/6: Processing titles since {YEAR_FILTER_THRESHOLD}...")
    # ... (same chunking logic for titles_df) ...
    filtered_chunks = []
    with pd.read_csv(BASICS_URL, sep='\t', usecols=['tconst', 'titleType', 'primaryTitle', 'genres', 'startYear'], chunksize=CHUNK_SIZE, compression='gzip') as reader:
        for chunk in reader:
            chunk['startYear'] = pd.to_numeric(chunk['startYear'], errors='coerce')
            chunk.dropna(subset=['startYear'], inplace=True)
            chunk = chunk[(chunk['startYear'] >= YEAR_FILTER_THRESHOLD) & (chunk['titleType'].isin(['movie', 'tvSeries']))]
            if not chunk.empty: filtered_chunks.append(chunk)
    titles_df = pd.concat(filtered_chunks, ignore_index=True)

    print(f"Step 3/6: Filtering ratings...")
    # ... (same chunking logic for ratings_df) ...
    relevant_ids = set(titles_df['tconst'])
    rating_chunks = []
    with pd.read_csv(RATINGS_URL, sep='\t', chunksize=CHUNK_SIZE, compression='gzip') as reader:
        for chunk in reader:
            rating_chunks.append(chunk[chunk['tconst'].isin(relevant_ids)])
    ratings_df = pd.concat(rating_chunks, ignore_index=True)

    print("Step 4/6: Merging and enriching final data...")
    titles_df['is_indian'] = titles_df['tconst'].isin(indian_ids)
    enriched_df = pd.merge(titles_df, ratings_df, on='tconst', how='left')
    enriched_df.fillna({'genres': '', 'averageRating': 0, 'numVotes': 0}, inplace=True)
    qualified_df = enriched_df[enriched_df['numVotes'] >= MINIMUM_VOTES_THRESHOLD].copy()
    qualified_df['metadata_soup'] = (qualified_df['primaryTitle'] + ' ' + qualified_df['genres'].str.replace(',', ' ')).fillna('')
    print(f"Final qualified titles for ML model: {len(qualified_df)}")

    # --- Step 5: Pre-compute TF-IDF only ---
    print("Step 5/6: Computing and saving TF-IDF vectorizer and sparse matrix...")
    if qualified_df.empty:
        print("No qualified titles found, exiting.")
        return
        
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    # This creates the sparse matrix, which is memory-efficient
    tfidf_matrix = tfidf_vectorizer.fit_transform(qualified_df['metadata_soup'])
    
    # --- Step 6: Persist Artifacts for the Live App ---
    print("Step 6/6: Saving computed artifacts to disk...")
    qualified_df.reset_index(drop=True, inplace=True)
    
    # CRITICAL: We now save the vectorizer and the sparse matrix, NOT cosine_sim.
    with open(os.path.join(ARTIFACTS_DIR, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    with open(os.path.join(ARTIFACTS_DIR, 'tfidf_matrix.pkl'), 'wb') as f:
        pickle.dump(tfidf_matrix, f)
        
    # We still need the DataFrame for mapping and metadata
    qualified_df.to_pickle(os.path.join(ARTIFACTS_DIR, 'enriched_titles.pkl'))
    
    # And the SQLite DB for any other potential uses
    db_df = qualified_df[['tconst', 'primaryTitle', 'titleType', 'is_indian']].rename(columns={'tconst': 'id', 'primaryTitle': 'name', 'titleType': 'type'})
    engine = create_engine(f'sqlite:///{os.path.join(ARTIFACTS_DIR, "addon_data.db")}')
    db_df.to_sql('titles', con=engine, if_exists='replace', index=False)
    
    end_time = time.time()
    print(f"--- Full build finished in {end_time - start_time:.2f} seconds. ---")

if __name__ == '__main__':
    build_artifacts()
