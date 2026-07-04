# Goodreads Crawler & Ranker WIP

Find books that match your tastes with this async Goodreads crawler + ML recommendation pipeline.

## Getting Started

### 0. **Populate Your Goodreads Library:**

<img src="assets/button.png" alt="alt text" width="300">

Your library seeds the crawler to find new similar books.<br>
Any books you've rated will train your personal model.

### 1. **Pip Install Requirements:**

```bash
pip install -r requirements.txt
playwright install chromium
nbstripout --install
nbdime config-git --enable
```

### 2. **Configure Log-in Credentials:**

Create a `.env` file in the root with:

```ini
GOODREADS_EMAIL=your_email@example.com
GOODREADS_PASSWORD=your_password
```

### 3. **Set up Ollama:**

#### A. Download and install [Ollama](https://ollama.com/download)

```bash
brew install ollama
```

#### B. Pull your desired embedding model. This repo uses [qwen3-embedding:8b](https://ollama.com/library/qwen3-embedding:8b) by default:

```bash
ollama pull qwen3-embedding:8b
```

### 4. **Run the recomendation pipeline:**

```bash
python3 cli.py run_pipeline
```

Crawling book profiles and embedding books descriptions can take _several hours_ when run for the first time.

### 5. **Review the results** (WIP\*):

Predictions are stored in `data/goodreads.db` under the `predictions` table.

You may join it with the `books` table and sort by one of the provided rating columns to view the best recommendations.
