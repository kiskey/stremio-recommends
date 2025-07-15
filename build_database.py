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
BASICS_URL = "https://datasets.imdbws.com/title.basics.tsv.gz"
AKAS_URL = "https://datasets.imdbws.com/title.akas.tsv.gz"
RATINGS_URL = "https://datasets.imdbws.com/title.ratings.tsv.gz"
ARTIFACTS_DIR = "artifacts"

def build_artifacts():
    """
    Performs a full build of the database and ML models from the latest IMDb data.
    """
    start_time = time.time()
    print("--- Starting Full Artifact Build ---")

    # Create artifacts directory if it doesn't exist
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    print(f"Artifacts will be saved in '{ARTIFACTS_DIR}/'")

    # --- EXTRACT ---
    print("Step 1/5: Downloading and reading datasets...")
    basics_df = pd.read_csv(BASICS_URL, sep='\t', low_memory=False, compression='gzip',
                            usecols=['tconst', 'titleType', 'primaryTitle', 'genres'])
    akas_df = pd.read_csv(AKAS_URL, sep='\t', low_memory=False, compression='gzip',
                          usecols=['titleId', 'region'])
    ratings_df = pd.read_csv(RATINGS_URL, sep='\t', low_memory=False, compression='gzip')
    print("Datasets loaded.")

    # --- TRANSFORM ---
    print("Step 2/5: Transforming and enriching data...")
    titles_df = basics_df[basics_df['titleType'].isin(['movie', 'tvSeries'])].copy()
    
    indian_akas = akas_df[akas_df['region'] == 'IN']
    indian_ids = set(indian_akas['titleId'].unique())
    titles_df['is_indian'] = titles_df['tconst'].isin(indian_ids)
    
    enriched_df = pd.merge(titles_df, ratings_df, on='tconst', how='left')
    
    enriched_df['genres'] = enriched_df['genres'].replace('\\N', '')
    enriched_df['averageRating'] = pd.to_numeric(enriched_df['averageRating'], errors='coerce').fillna(0)
    enriched_df['numVotes'] = pd.to_numeric(enriched_df['numVotes'], errors='coerce').fillna(0)
    
    qualified_df = enriched_df[enriched_df['numVotes'] >= MINIMUM_VOTES_THRESHOLD].copy()
    qualified_df['metadata_soup'] = (qualified_df['primaryTitle'] + ' ' + qualified_df['genres'].str.replace(',', ' ')).fillna('')
    print(f"Data transformed. Total qualified titles for ML model: {len(qualified_df)}")

    # --- MACHINE LEARNING ---
    print("Step 3/5: Computing TF-IDF and Cosine Similarity Matrix...")
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(qualified_df['metadata_soup'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print("ML matrix computed.")

    # --- PERSIST ARTIFACTS ---
    print("Step 4/5: Saving computed artifacts to disk...")
    qualified_df.reset_index(drop=True, inplace=True)
    
    # Save ML artifacts
    with open(os.path.join(ARTIFACTS_DIR, 'cosine_sim.pkl'), 'wb') as f:
        pickle.dump(cosine_sim, f)
    qualified_df.to_pickle(os.path.join(ARTIFACTS_DIR, 'enriched_titles.pkl'))
    
    # Save SQLite DB
    db_df = qualified_df[['tconst', 'primaryTitle', 'titleType', 'is_indian']].rename(columns={'tconst': 'id', 'primaryTitle': 'name', 'titleType': 'type'})
    engine = create_engine(f'sqlite:///{os.path.join(ARTIFACTS_DIR, "addon_data.db")}')
    db_df.to_sql('titles', con=engine, if_exists='replace', index=False)
    print("All artifacts saved successfully.")
    
    end_time = time.time()
    print(f"--- Full build finished in {end_time - start_time:.2f} seconds. ---")

if __name__ == '__main__':
    build_artifacts()
