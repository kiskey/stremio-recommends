# build_database.py

import pandas as pd
from sqlalchemy import create_engine
import pickle
import time
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# --- CONFIGURATION ---
MINIMUM_VOTES_THRESHOLD = 500
YEAR_FILTER_THRESHOLD = 1980
CHUNK_SIZE = 250000
ARTIFACTS_DIR = "artifacts"

# Data source URLs
BASICS_URL = "https://datasets.imdbws.com/title.basics.tsv.gz"
AKAS_URL = "https://datasets.imdbws.com/title.akas.tsv.gz"
RATINGS_URL = "https://datasets.imdbws.com/title.ratings.tsv.gz"
PRINCIPALS_URL = "https://datasets.imdbws.com/title.principals.tsv.gz"
NAMES_URL = "https://datasets.imdbws.com/name.basics.tsv.gz"

def build_artifacts():
    start_time = time.time()
    print("--- Starting Advanced Artifact Build (Concept-Based & Memory-Optimized) ---")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # --- Step 1: Identify all Indian titles from the AKAs file ---
    print("Step 1/8: Identifying all Indian titles...")
    indian_ids = set()
    with pd.read_csv(AKAS_URL, sep='\t', usecols=['titleId', 'region'], chunksize=CHUNK_SIZE, compression='gzip') as reader:
        for chunk in reader:
            indian_ids.update(chunk[chunk['region'] == 'IN']['titleId'])
    print(f"Found {len(indian_ids)} unique titles associated with India.")

    # --- Step 2: Build a filtered list of modern movies and series from the Basics file ---
    print(f"Step 2/8: Processing titles since {YEAR_FILTER_THRESHOLD} in chunks...")
    filtered_chunks = []
    with pd.read_csv(BASICS_URL, sep='\t', usecols=['tconst', 'titleType', 'primaryTitle', 'genres', 'startYear'], chunksize=CHUNK_SIZE, compression='gzip') as reader:
        for chunk in reader:
            chunk['startYear'] = pd.to_numeric(chunk['startYear'], errors='coerce')
            chunk.dropna(subset=['startYear'], inplace=True)
            chunk['startYear'] = chunk['startYear'].astype(int)
            filtered_chunk = chunk[
                (chunk['startYear'] >= YEAR_FILTER_THRESHOLD) &
                (chunk['titleType'].isin(['movie', 'tvSeries']))
            ]
            if not filtered_chunk.empty:
                filtered_chunks.append(filtered_chunk)
    titles_df = pd.concat(filtered_chunks, ignore_index=True)
    print(f"Found {len(titles_df)} movies/series since {YEAR_FILTER_THRESHOLD}.")

    # --- Step 3: Pre-filter the ratings file for relevant titles ---
    relevant_ids = set(titles_df['tconst'])
    print(f"Step 3/8: Filtering ratings for {len(relevant_ids)} relevant titles...")
    rating_chunks = []
    with pd.read_csv(RATINGS_URL, sep='\t', chunksize=CHUNK_SIZE, compression='gzip') as reader:
        for chunk in reader:
            rating_chunks.append(chunk[chunk['tconst'].isin(relevant_ids)])
    ratings_df = pd.concat(rating_chunks, ignore_index=True)
    print(f"Found {len(ratings_df)} ratings for relevant titles.")

    # --- Step 4: Process Principals to get Directors and top 3 Actors ---
    print("Step 4/8: Processing directors and actors...")
    # Load name mappings first - this file is small enough to load fully
    names_df = pd.read_csv(NAMES_URL, sep='\t', usecols=['nconst', 'primaryName'])
    
    # Process principals in chunks
    principal_chunks = []
    with pd.read_csv(PRINCIPALS_URL, sep='\t', usecols=['tconst', 'ordering', 'nconst', 'category'], chunksize=CHUNK_SIZE*2) as reader:
        for chunk in reader:
            filtered_chunk = chunk[
                chunk['tconst'].isin(relevant_ids) & 
                chunk['category'].isin(['director', 'actor']) &
                (chunk['ordering'] <= 3) # Directors are usually ordering 1-2, this gets top 3 actors
            ]
            if not filtered_chunk.empty:
                principal_chunks.append(filtered_chunk)
    
    if principal_chunks:
        principals_df = pd.concat(principal_chunks, ignore_index=True)
        principals_df = pd.merge(principals_df, names_df, on='nconst', how='left').dropna(subset=['primaryName'])
        # Clean names to create single tokens, e.g., "ChristopherNolan"
        principals_df['primaryName'] = principals_df['primaryName'].str.replace(' ', '', regex=False)
        
        # Aggregate principals into a single string for each movie
        def aggregate_principals(df):
            directors = ' '.join(df[df['category'] == 'director']['primaryName'])
            actors = ' '.join(df[df['category'] == 'actor']['primaryName'])
            return pd.Series([directors, actors], index=['directors', 'actors'])

        aggregated_principals = principals_df.groupby('tconst').apply(aggregate_principals)
        titles_df = pd.merge(titles_df, aggregated_principals, on='tconst', how='left')
    else:
        titles_df['directors'] = ''
        titles_df['actors'] = ''
        
    titles_df.fillna({'directors': '', 'actors': ''}, inplace=True)

    # --- Step 5: Final Merge, Enrichment, and Qualification ---
    print("Step 5/8: Merging, enriching, and qualifying final data...")
    titles_df['is_indian'] = titles_df['tconst'].isin(indian_ids)
    enriched_df = pd.merge(titles_df, ratings_df, on='tconst', how='left')
    enriched_df.fillna({'genres': '', 'averageRating': 0, 'numVotes': 0}, inplace=True)
    qualified_df = enriched_df[enriched_df['numVotes'] >= MINIMUM_VOTES_THRESHOLD].copy()
    print(f"Final qualified titles for ML model: {len(qualified_df)}")

    # --- Step 6: Create the NEW, Weighted Metadata Soup ---
    print("Step 6/8: Constructing weighted metadata soup...")
    if not qualified_df.empty:
        # Give genres a weight of 3, directors a weight of 2, actors and title a weight of 1
        qualified_df['genres_weighted'] = qualified_df['genres'].str.replace(',', ' ', regex=False) + ' '
        qualified_df['genres_weighted'] = qualified_df['genres_weighted'] * 3
        
        qualified_df['directors_weighted'] = qualified_df['directors'] + ' '
        qualified_df['directors_weighted'] = qualified_df['directors_weighted'] * 2
        
        qualified_df['metadata_soup'] = qualified_df['genres_weighted'] + \
                                          qualified_df['directors_weighted'] + \
                                          qualified_df['actors'] + ' ' + \
                                          qualified_df['primaryTitle']
    else:
        print("No qualified titles found, skipping ML and artifact generation.")
        return

    # --- Step 7: Train the Phrase-Based TF-IDF Model ---
    print("Step 7/8: Computing phrase-based TF-IDF...")
    # ngram_range=(1, 2) considers both single words AND two-word phrases
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(qualified_df['metadata_soup'])

    # --- Step 8: Persist Artifacts ---
    print("Step 8/8: Saving computed artifacts to disk...")
    qualified_df.reset_index(drop=True, inplace=True)
    
    with open(os.path.join(ARTIFACTS_DIR, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    with open(os.path.join(ARTIFACTS_DIR, 'tfidf_matrix.pkl'), 'wb') as f:
        pickle.dump(tfidf_matrix, f)
        
    qualified_df.to_pickle(os.path.join(ARTIFACTS_DIR, 'enriched_titles.pkl'))
    
    db_df = qualified_df[['tconst', 'primaryTitle', 'titleType', 'is_indian']].rename(columns={'tconst': 'id', 'primaryTitle': 'name', 'titleType': 'type'})
    engine = create_engine(f'sqlite:///{os.path.join(ARTIFACTS_DIR, "addon_data.db")}')
    db_df.to_sql('titles', con=engine, if_exists='replace', index=False)
    
    end_time = time.time()
    print(f"--- Full build finished in {end_time - start_time:.2f} seconds. ---")

if __name__ == '__main__':
    build_artifacts()
