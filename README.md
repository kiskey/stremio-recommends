# Stremio "For You" Recommendation Addon

A self-hosted Stremio addon that provides personalized movie and series recommendations based on your viewing history. It intelligently seeds recommendations, sorts them by relevance and rating, and presents them in separate, high-quality catalogs for an integrated and polished user experience.

This addon is designed to be fully automated, using GitHub Actions to build fresh data artifacts and a self-contained Docker image on a schedule.

## Key Features

-   **Truly Personalized Recommendations:** Learns from the movies and series you watch to suggest similar content.
-   **Separate, Organized Catalogs:** Provides two distinct catalogs: "Recommended Movies" and "Recommended Series."
-   **Intelligent Seeding:** Each catalog is seeded from its own specific media type history.
-   **Configurable Region Sorting:** Recommendations are sorted by a user-defined priority list of regions (e.g., Indian first, then US, then UK), followed by all other international content.
-   **Pagination Support:** Both catalogs are fully paginated to handle large recommendation lists.
-   **Configurable Page Size:** The number of items per page can be easily configured.
-   **Concept-Based Matching:** The machine learning model prioritizes **genres, directors, and top actors** over simple title keywords.
-   **Fully Automated & Self-Contained:** A GitHub Actions workflow automatically rebuilds the data and the Docker image every 3 days.

## Technology Stack
- **Backend:** Python 3.9, Flask, Gunicorn
- **Data Processing & ML:** Pandas, Scikit-learn, NumPy
- **CI/CD & Deployment:** Docker, Docker Compose, GitHub Actions
- **Data Source:** IMDb Datasets

## Configuration (Environment Variables)

You can customize the addon's behavior by setting environment variables in your `docker-compose.yml` file.

-   **`PRIORITY_REGIONS`**: A comma-separated, ordered list of [ISO 3166-1 alpha-2 country codes](https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes) to prioritize in recommendations. This same variable should be set in the GitHub Actions workflow file to ensure data is built correctly.
    -   *Example*: `IN,US,UK` will show Indian content first, then US content, then UK content, followed by all other international content.
    -   *Default*: `IN`

-   **`PAGE_SIZE`**: The number of items to show per catalog page. This also controls the `skip` interval for pagination.
    -   *Example*: `30`
    -   *Default*: `50`

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
