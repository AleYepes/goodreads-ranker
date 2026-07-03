# Goodreads Crawler & Ranker WIP

Async crawler and ML recommendation pipeline to find good books to read.

## Setup 
1. **Save Books to Your Goodreads Library:**

    <img src="assets/button.png" alt="alt text" width="300">

    Your library seeds the crawler.<br>
    Books marked as 'Read' train your personal model.

2. **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    playwright install chromium
    nbstripout --install
    nbdime config-git --enable
    ```

3. **Configure Credentials:**
    
    Create a `.env` file in the root with:
    ```ini
    GOODREADS_EMAIL=your_email@example.com
    GOODREADS_PASSWORD=your_password
    ```

4. **Set up Ollama** *(If you want to train a personal model)*:

    a. Download and install [Ollama](https://ollama.com/download)
    ```bash
    brew install ollama
    ```
    b. Pull an embedding model. I'm using [qwen3 0.6b](https://ollama.com/library/qwen3-embedding:0.6b):
    ```bash
    ollama pull qwen3-embedding:0.6b
    ```
    c. Run Ollama to serve inference:
    ```bash
    ollama serve
    ```


## Usage
1. **Run the main pipeline:**
    ```bash
    python3 cli.py run_pipeline
    ```
    By default this initializes SQLite, seeds your library and friend ratings,
    crawls missing seed books, embeds books that need embeddings, ranks with
    stored/default model parameters, and prints a verification summary.

2. **Run individual stages when needed:**
    ```bash
    python3 cli.py seed
    python3 cli.py crawl --limit=25
    python3 cli.py embed
    python3 cli.py rank
    python3 cli.py verify
    ```

    `crawl --limit=None` crawls missing seed books only. Positive limits crawl
    seed books first and then expansion books up to the limit. Zero or negative
    limits crawl indefinitely, including expansion books.
