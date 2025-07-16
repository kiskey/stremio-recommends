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
# NEW: Read priority regions from environment variable, default to 'IN'
PRIORITY_REGIONS = os.environ.get('PRIORITY_REGIONS', 'IN').split(',')

# Data source URLs
BASICS_URL = "https://datasets.imdbws.com/title.basics.tsv.gz"
AKAS_URL = "https://datasets.imdbws.com/title.akas.tsv.gz"
RATINGS_URL = "https://datasets.imdbws.com/title.ratings.tsv.gz"
PRINCIPALS_URL = "https://datasets.imdbws.com/title.principals.tsv.gz"
NAMES_URL = "https://datasets.imdbws.com/name.basics.tsv.gz"

def get_primary_region(group):
    """For a group of regions for a single title, find the first one that matches our priority list."""
    for region in PRIORITY_REGIONS:
        if region in group.values:
            return region
    return 'Other' # Fallback for content not in our priority list

def build_artifacts():
    start_time = time.time()
    print(f"--- Building artifacts with region priority: {PRIORITY_REGIONS} ---")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # Step 1 & 2: Get modern movies/series titles
    print(f"Step 1/8: Processing titles since {YEAR_FILTER_THRESHOLD} in chunks...")
    filtered_chunks = []
    with pd.read_csv(BASICS_URL, sep='\t', usecols=['tconst', 'titleType', 'primaryTitle', 'genres', 'startYear'], chunksize=CHUNK_SIZE) as reader:
        for chunk in reader:
            chunk['startYear'] = pd.to_numeric(chunk['startYear'], errors='coerce')
            chunk.dropna(subset=['startYear'], inplace=True)
            chunk['startYear'] = chunk['startYear'].astype(int)
            filtered_chunk = chunk[(chunk['startYear'] >= YEAR_FILTER_THRESHOLD) & (chunk['titleType'].isin(['movie', 'tvSeries']))]
            if not filtered_chunk.empty:
                filtered_chunks.append(filtered_chunk)
    titles_df = pd.concat(filtered_chunks, ignore_index=True)
    print(f"Found {len(titles_df)} movies/series since {YEAR_FILTER_THRESHOLD}.")

    relevant_ids = set(titles_df['tconst'])

    # Step 3: NEW - Process AKAs to determine primary region
    print("Step 2/8: Determining primary region for all titles...")
    aka_chunks = []
    with pd.read_csv(AKAS_URL, sep='\t', usecols=['titleId', 'region'], chunksize=CHUNK_SIZE*2) as reader:
        for chunk in reader:
            aka_chunks.append(chunk[chunk['titleId'].isin(relevant_ids)].dropna())
    
    if aka_chunks:
        akas_df = pd.concat(aka_chunks, ignore_index=True)
        region_map = akas_df.groupby('titleId')['region'].apply(get_primary_region).rename('primary_region')
        titles_df = pd.merge(titles_df, region_map, left_on='tconst', right_index=True, how='left')
        titles_df['primary_region'].fillna('Other', inplace=True)
    else:
        titles_df['primary_region'] = 'Other'

    # The rest of the steps proceed as before, now with the 'primary_region' column
    print(f"Step 3/8: Filtering ratings for {len(relevant_ids)} relevant titles...")
    rating_chunks = []
    with pd.read_csv(RATINGS_URL, sep='\t', chunksize=CHUNK_SIZE) as reader:
        for chunk in reader:
            rating_chunks.append(chunk[chunk['tconst'].isin(relevant_ids)])
    ratings_df = pd.concat(rating_chunks, ignore_index=True)

    print("Step 4/8: Processing directors and actors...")
    names_df = pd.read_csv(NAMES_URL, sep='\t', usecols=['nconst', 'primaryName'])
    principal_chunks = []
    with pd.read_csv(PRINCIPALS_URL, sep='\t', usecols=['tconst', 'ordering', 'nconst', 'category'], chunksize=CHUNK_SIZE*2) as reader:
        for chunk in reader:
            filtered_chunk = chunk[chunk['tconst'].isin(relevant_ids) & chunk['category'].isin(['director', 'actor']) & (chunk['ordering'] <= 3)]
            if not filtered_chunk.empty:
                principal_chunks.append(filtered_chunk)
    if principal_chunks:
        principals_df = pd.concat(principal_chunks, ignore_index=True)
        principals_df = pd.merge(principals_df, names_df, on='nconst', how='left').dropna(subset=['primaryName'])
        principals_df['primaryName'] = principals_df['primaryName'].str.replace(' ', '', regex=False)
        aggregated_principals = principals_df.groupby('tconst').apply(lambda df: pd.Series([' '.join(df[df['category'] == 'director']['primaryName']), ' '.join(df[df['category'] == 'actor']['primaryName'])], index=['directors', 'actors']))
        titles_df = pd.merge(titles_df, aggregated_principals, on='tconst', how='left')
    else:
        titles_df['directors'], titles_df['actors'] = '', ''
    titles_df.fillna({'directors': '', 'actors': ''}, inplace=True)

    print("Step 5/8: Merging, enriching, and qualifying final data...")
    enriched_df = pd.merge(titles_df, ratings_df, on='tconst', how='left')
    enriched_df.fillna({'genres': '', 'averageRating': 0, 'numVotes': 0}, inplace=True)
    qualified_df = enriched_df[enriched_df['numVotes'] >= MINIMUM_VOTES_THRESHOLD].copy()
    print(f"Final qualified titles for ML model: {len(qualified_df)}")

    if qualified_df.empty:
        print("No qualified titles found, skipping ML and artifact generation.")
        return

    print("Step 6/8: Constructing weighted metadata soup...")
    qualified_df['genres_weighted'] = (qualified_df['genres'].str.replace(',', ' ', regex=False) + ' ') * 3
    qualified_df['directors_weighted'] = (qualified_df['directors'] + ' ') * 2
    qualified_df['metadata_soup'] = qualified_df['genres_weighted'] + qualified_df['directors_weighted'] + qualified_df['actors'] + ' ' + qualified_df['primaryTitle']
    
    print("Step 7/8: Computing phrase-based TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(qualified_df['metadata_soup'])

    print("Step 8/8: Saving computed artifacts and version file...")
    qualified_df.reset_index(drop=True, inplace=True)
    with open(os.path.join(ARTIFACTS_DIR, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    with open(os.path.join(ARTIFACTS_DIR, 'tfidf_matrix.pkl'), 'wb') as f:
        pickle.dump(tfidf_matrix, f)
    qualified_df.to_pickle(os.path.join(ARTIFACTS_DIR, 'enriched_titles.pkl'))
    build_version = os.environ.get('GITHUB_RUN_NUMBER', 'local')
    with open(os.path.join(ARTIFACTS_DIR, 'version.txt'), 'w') as f:
        f.write(build_version)
    print(f"Artifacts built with version: {build_version}")

    end_time = time.time()
    print(f"--- Full build finished in {end_time - start_time:.2f} seconds. ---")

if __name__ == '__main__':
    build_artifacts()
