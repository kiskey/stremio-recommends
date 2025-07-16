# build_database.py

import pandas as pd
from sqlalchemy import create_engine
import pickle
import time
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
MINIMUM_VOTES_THRESHOLD = 500
YEAR_FILTER_THRESHOLD = 1980
CHUNK_SIZE = 100000  # Process 100,000 rows at a time
BASICS_URL = "https://datasets.imdbws.com/title.basics.tsv.gz"
AKAS_URL = "https://datasets.imdbws.com/title.akas.tsv.gz"
RATINGS_URL = "https://datasets.imdbws.com/title.ratings.tsv.gz"
ARTIFACTS_DIR = "artifacts"

def build_artifacts():
    start_time = time.time()
    print("--- Starting Full Artifact Build (Memory-Optimized) ---")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # --- Step 1: Get all Indian Title IDs by chunking the large AKAs file ---
    print("Step 1/6: Identifying all Indian titles...")
    indian_ids = set()
    with pd.read_csv(AKAS_URL, sep='\t', usecols=['titleId', 'region'], chunksize=CHUNK_SIZE, compression='gzip') as reader:
        for chunk in reader:
            indian_chunk = chunk[chunk['region'] == 'IN']
            indian_ids.update(indian_chunk['titleId'])
    print(f"Found {len(indian_ids)} unique titles associated with India.")

    # --- Step 2: Process basics file in chunks to build our core DataFrame ---
    print("Step 2/6: Processing titles in chunks...")
    filtered_chunks = []
    with pd.read_csv(BASICS_URL, sep='\t', usecols=['tconst', 'titleType', 'primaryTitle', 'genres', 'startYear'], chunksize=CHUNK_SIZE, compression='gzip') as reader:
        for chunk in reader:
            chunk.dropna(subset=['startYear'], inplace=True)
            chunk['startYear'] = pd.to_numeric(chunk['startYear'], errors='coerce')
            chunk = chunk[
                (chunk['startYear'] >= YEAR_FILTER_THRESHOLD) &
                (chunk['titleType'].isin(['movie', 'tvSeries']))
            ]
            if not chunk.empty:
                filtered_chunks.append(chunk)
    
    titles_df = pd.concat(filtered_chunks, ignore_index=True)
    print(f"Processed and filtered down to {len(titles_df)} titles since {YEAR_FILTER_THRESHOLD}.")

    # --- Step 3: Load ratings and enrich the DataFrame ---
    print("Step 3/6: Enriching data with ratings and classifications...")
    ratings_df = pd.read_csv(RATINGS_URL, sep='\t')
    
    titles_df['is_indian'] = titles_df['tconst'].isin(indian_ids)
    enriched_df = pd.merge(titles_df, ratings_df, on='tconst', how='left')
    
    enriched_df['genres'] = enriched_df['genres'].replace('\\N', '')
    enriched_df['averageRating'] = pd.to_numeric(enriched_df['averageRating'], errors='coerce').fillna(0)
    enriched_df['numVotes'] = pd.to_numeric(enriched_df['numVotes'], errors='coerce').fillna(0)
    
    qualified_df = enriched_df[enriched_df['numVotes'] >= MINIMUM_VOTES_THRESHOLD].copy()
    qualified_df['metadata_soup'] = (qualified_df['primaryTitle'] + ' ' + qualified_df['genres'].str.replace(',', ' ')).fillna('')
    print(f"Data transformed. Total qualified titles for ML model: {len(qualified_df)}")

    # --- Step 4: Machine Learning ---
    print("Step 4/6: Computing TF-IDF and Cosine Similarity Matrix...")
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(qualified_df['metadata_soup'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print("ML matrix computed.")

    # --- Step 5: Persist Artifacts ---
    print("Step 5/6: Saving computed artifacts to disk...")
    qualified_df.reset_index(drop=True, inplace=True)
    
    with open(os.path.join(ARTIFACTS_DIR, 'cosine_sim.pkl'), 'wb') as f:
        pickle.dump(cosine_sim, f)
    qualified_df.to_pickle(os.path.join(ARTIFACTS_DIR, 'enriched_titles.pkl'))
    
    db_df = qualified_df[['tconst', 'primaryTitle', 'titleType', 'is_indian']].rename(columns={'tconst': 'id', 'primaryTitle': 'name', 'titleType': 'type'})
    engine = create_engine(f'sqlite:///{os.path.join(ARTIFACTS_DIR, "addon_data.db")}')
    db_df.to_sql('titles', con=engine, if_exists='replace', index=False)
    print("All artifacts saved successfully.")
    
    end_time = time.time()
    print(f"--- Full build finished in {end_time - start_time:.2f} seconds. ---")

if __name__ == '__main__':
    build_artifacts()
