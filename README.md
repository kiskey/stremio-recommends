# Stremio "For You" Recommendation Addon

A self-hosted Stremio addon that provides personalized movie and series recommendations based on your viewing history. It intelligently seeds recommendations, sorts them by relevance and rating, and presents them in separate, high-quality catalogs for an integrated and polished user experience.

This addon is designed to be fully automated, using GitHub Actions to build fresh data artifacts and a self-contained Docker image on a schedule.

## Key Features

-   **Truly Personalized Recommendations:** Learns from the movies and series you watch to suggest similar content.
-   **Separate, Organized Catalogs:** Provides two distinct catalogs on your Stremio home screen: "Recommended Movies" and "Recommended Series."
-   **Intelligent Seeding:** The movie catalog is seeded from your movie history, and the series catalog is seeded from your series history, ensuring relevant suggestions.
-   **Smart Sorting:** Recommendations are sorted first by region (Indian content prioritized), then by relevance score and average rating.
-   **Concept-Based Matching:** The machine learning model goes beyond simple keywords. It understands and prioritizes **genres, directors, and top actors**, leading to high-quality, thematic recommendations.
-   **Full Poster Support:** Catalog views are fully populated with posters for a rich user experience, without slowing down the addon.
-   **Fully Automated & Self-Contained:** A GitHub Actions workflow automatically rebuilds the data and the Docker image every 3 days, ensuring your recommendations stay fresh. The final Docker image includes all necessary data.
-   **Lightweight & Performant:** Designed to run efficiently in a low-resource environment (like a small VPS or Raspberry Pi) using a "split brain" architecture.

## How It Works (Architecture)

This project uses a modern architecture that separates the heavy data processing from the lightweight live application.

#### 1. Offline Build Pipeline (GitHub Actions)

-   **Trigger:** Runs on a schedule (every 3 days) or can be triggered manually.
-   **Data Fetching:** Downloads the latest datasets from IMDb (titles, ratings, names, etc.).
-   **Filtering & Enrichment:** Processes millions of records to create a clean, modern (post-1980) dataset with only qualified titles (minimum vote count). It enriches this data with director and actor information.
-   **ML Model Training:** Builds a "metadata soup" for each title, heavily weighting genres and creators. It then computes a TF-IDF sparse matrix from this data, which represents the "knowledge" of how titles relate to each other.
-   **Bundling:** The script creates a set of data artifacts (`enriched_titles.pkl`, `tfidf_matrix.pkl`, etc.).
-   **Docker Build:** A lean Docker image is built, copying the application code and the ~15MB of data artifacts directly into the image.
-   **Push to Registry:** The final, self-contained image is pushed to the GitHub Container Registry.

#### 2. Online Addon Server (Your Deployed Docker Container)

-   **Startup:** On startup, a pre-flight script verifies that all data artifacts are present before starting the web server. The Flask application loads the data artifacts into memory once.
-   **History Logging:** The addon listens silently as you browse Stremio, logging the IMDb IDs of movies and series you click on to a persistent `watch_history.db` file.
-   **On-Demand Recommendations:** When you open a "Recommended" catalog, the app:
    1.  Loads your recent viewing history for that specific media type (e.g., only movies).
    2.  Performs a series of lightning-fast `cosine_similarity` calculations between your history and the entire knowledge base.
    3.  Generates a pool of recommendations, filters, sorts them by region and rating, and serves the final list to Stremio.

## Technology Stack

-   **Backend:** Python 3.9, Flask, Gunicorn
-   **Data Processing & ML:** Pandas, Scikit-learn, NumPy
-   **CI/CD & Deployment:** Docker, Docker Compose, GitHub Actions
-   **Data Source:** IMDb Datasets

## Deployment Guide

This guide will walk you through deploying your own instance of the addon.

#### Prerequisites

1.  A server with Docker and Docker Compose installed.
2.  A GitHub account.

#### Step 1: Fork the Repository

Fork this repository to your own GitHub account. The GitHub Actions workflow will run under your account.

#### Step 2: Configure GitHub Actions Permissions

The automated workflow needs permission to push the Docker image to your account's registry.

1.  In your forked repository, go to `Settings` > `Actions` > `General`.
2.  Scroll down to **Workflow permissions**.
3.  Ensure that **"Read and write permissions"** is selected. This allows the workflow to be granted `packages: write` scope.
4.  Save the changes.

#### Step 3: Run the GitHub Action

Navigate to the "Actions" tab in your repository, select the "Build and Push Self-Contained Docker Image" workflow, and click "Run workflow" to trigger your first build. This will create the initial Docker image.

#### Step 4: Server Setup and Deployment

1.  SSH into your server.

2.  Create a dedicated directory for your addon.
    ```bash
    mkdir ~/stremio-addon
    cd ~/stremio-addon
    ```

3.  Create the `docker-compose.yml` file in this directory.
    ```bash
    nano docker-compose.yml
    ```
    Paste the following content into the file. **Remember to replace `YOUR_GITHUB_USERNAME/YOUR_REPO_NAME` with your actual GitHub username and repository name.**

    ```yaml
    # docker-compose.yml
    version: '3.8'

    services:
      stremio-recommends:
        image: ghcr.io/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME:latest
        container_name: stremio-recommends-addon
        volumes:
          # This volume persists your watch history across container updates.
          - ./watch_history.db:/usr/src/app/artifacts/watch_history.db
        ports:
          - "8080:5000"
        restart: always
    ```

4.  Create an empty placeholder file for the history database. This ensures the volume mount works correctly on the first run.
    ```bash
    touch watch_history.db
    ```

5.  Pull the latest image from the registry and start the addon.
    ```bash
    docker-compose pull
    docker-compose up -d
    ```

## Usage

1.  Find your server's public IP address or domain name.
2.  The manifest URL for your addon will be: `http://<YOUR_SERVER_IP>:8080/manifest.json`
3.  Open Stremio, go to the Addons page, and paste the manifest URL into the search bar.
4.  Click "Install."

Your new "Recommended Movies" and "Recommended Series" catalogs will now appear on your Stremio home screen. They will initially be empty. As you watch content, they will begin to populate with personalized suggestions.

## Maintenance and Updates

-   **Data Updates:** The GitHub Action is scheduled to automatically rebuild the data and the Docker image every 3 days.
-   **Application Updates:** To update your deployed addon to the latest version, simply run the following commands on your server:
    ```bash
    cd ~/stremio-addon
    docker-compose pull
    docker-compose up -d
    ```

## License

This project is licensed under the MIT License.
