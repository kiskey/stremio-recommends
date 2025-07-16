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
CHUNK_SIZE = 250000  # Process rows in chunks to keep memory usage low
BASICS_URL = "https://datasets.imdbws.com/title.basics.tsv.gz"
AKAS_URL = "https://datasets.imdbws.com/title.akas.tsv.gz"
RATINGS_URL = "https://datasets.imdbws.com/title.ratings.tsv.gz"
ARTIFACTS_DIR = "artifacts"

def build_artifacts():
    start_time = time.time()
    print("--- Starting Full Artifact Build (Ultra Memory-Optimized) ---")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # --- Step 1: Identify all Indian titles by chunking the large AKAs file ---
    print("Step 1/7: Identifying all Indian titles...")
    indian_ids = set()
    with pd.read_csv(AKAS_URL, sep='\t', usecols=['titleId', 'region'], chunksize=CHUNK_SIZE, compression='gzip') as reader:
        for chunk in reader:
            indian_chunk = chunk[chunk['region'] == 'IN']
            indian_ids.update(indian_chunk['titleId'])
    print(f"Found {len(indian_ids)} unique titles associated with India.")

    # --- Step 2: Build a filtered list of modern movies and series ---
    # This is a critical memory optimization. We build a DataFrame of only the titles we need,
    # chunk by chunk, instead of loading the whole file.
    print(f"Step 2/7: Processing titles since {YEAR_FILTER_THRESHOLD} in chunks...")
    filtered_chunks = []
    with pd.read_csv(BASICS_URL, sep='\t', usecols=['tconst', 'titleType', 'primaryTitle', 'genres', 'startYear'], chunksize=CHUNK_SIZE, compression='gzip') as reader:
        for chunk in reader:
            # Coerce year to numeric and drop rows with invalid year data
            chunk['startYear'] = pd.to_numeric(chunk['startYear'], errors='coerce')
            chunk.dropna(subset=['startYear'], inplace=True)
            chunk['startYear'] = chunk['startYear'].astype(int)

            # Apply our core filters directly to the chunk
            filtered_chunk = chunk[
                (chunk['startYear'] >= YEAR_FILTER_THRESHOLD) &
                (chunk['titleType'].isin(['movie', 'tvSeries']))
            ]
            if not filtered_chunk.empty:
                filtered_chunks.append(filtered_chunk)
    
    # Concatenate the small, pre-filtered chunks into our main titles DataFrame
    titles_df = pd.concat(filtered_chunks, ignore_index=True)
    print(f"Found {len(titles_df)} movies/series since {YEAR_FILTER_THRESHOLD}.")

    # --- Step 3: Pre-filter the ratings file ---
    # Get a set of IDs we actually need ratings for. This is much smaller than all IDs.
    relevant_ids = set(titles_df['tconst'])
    print(f"Step 3/7: Filtering ratings for {len(relevant_ids)} relevant titles...")
    
    rating_chunks = []
    with pd.read_csv(RATINGS_URL, sep='\t', chunksize=CHUNK_SIZE, compression='gzip') as reader:
        for chunk in reader:
            # Keep only the rows from the chunk that match our relevant IDs
            relevant_ratings = chunk[chunk['tconst'].isin(relevant_ids)]
            if not relevant_ratings.empty:
                rating_chunks.append(relevant_ratings)
    
    ratings_df = pd.concat(rating_chunks, ignore_index=True)
    print(f"Found {len(ratings_df)} ratings for relevant titles.")

    # --- Step 4: Enrich the DataFrame with final classifications and ratings ---
    print("Step 4/7: Merging and enriching final data...")
    titles_df['is_indian'] = titles_df['tconst'].isin(indian_ids)
    enriched_df = pd.merge(titles_df, ratings_df, on='tconst', how='left')
    
    enriched_df['genres'] = enriched_df['genres'].replace('\\N', '')
    enriched_df['averageRating'] = pd.to_numeric(enriched_df['averageRating'], errors='coerce').fillna(0)
    enriched_df['numVotes'] = pd.to_numeric(enriched_df['numVotes'], errors='coerce').fillna(0)
    
    # --- Step 5: Final Qualification and ML Prep ---
    print("Step 5/7: Applying vote threshold and preparing for ML...")
    qualified_df = enriched_df[enriched_df['numVotes'] >= MINIMUM_VOTES_THRESHOLD].copy()
    qualified_df['metadata_soup'] = (qualified_df['primaryTitle'] + ' ' + qualified_df['genres'].str.replace(',', ' ')).fillna('')
    print(f"Final qualified titles for ML model: {len(qualified_df)}")

    # --- Step 6: Machine Learning ---
    print("Step 6/7: Computing TF-IDF and Cosine Similarity Matrix...")
    if qualified_df.empty:
        print("No qualified titles found, skipping ML and artifact generation.")
        return
        
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(qualified_df['metadata_soup'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print("ML matrix computed.")

    # --- Step 7: Persist Artifacts ---
    print("Step 7/7: Saving computed artifacts to disk...")
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
